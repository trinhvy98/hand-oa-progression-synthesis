"""Clinical image quality filtering for raw finger joint radiographs (Filter 1)."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class QualityFilterConfig:
    """Clinical Filter 1 thresholds removing truly corrupt, blurred, and off-center radiographs."""

    # Criterion 1: Blurred / low contrast / obscured central joint / X-ray saturation
    min_laplacian_var: float = 18.0
    min_norm_fg_laplacian_var: float = 95.0
    min_dynamic_range: float = 60.0
    max_overexposed_frac: float = 0.15
    overexposed_pixel_thresh: int = 245
    min_center_mean: float = 24.0

    # Criterion 2: Joint cropped off-center or missing from central 40% articular box
    min_center_fg_frac: float = 0.55
    max_center_dx: float = 30.0
    max_center_dy: float = 32.0
    max_center_dist: float = 36.0

    # Criterion 3: Aspect distortion or empty/oversized bone silhouette
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 1.35
    min_bone_area_frac: float = 0.16
    max_bone_area_frac: float = 0.96

    # Criterion 4: Excess shaft tilt
    max_tilt_deg: float = 45.0

    # Criterion 5: Corrupt or blank images
    min_foreground_pixels: int = 10


def _otsu_mask(gray_u8: np.ndarray) -> np.ndarray:
    """Compute binary foreground mask using exact histogram Otsu thresholding."""
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = float(gray_u8.size)
    sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))
    w_b = np.cumsum(hist)
    w_f = total - w_b
    sum_b = np.cumsum(np.arange(256, dtype=np.float64) * hist)

    mu_b = np.divide(sum_b, w_b, out=np.zeros_like(sum_b), where=w_b > 0)
    mu_f = np.divide(sum_total - sum_b, w_f, out=np.zeros_like(sum_b), where=w_f > 0)
    valid = (w_b > 0) & (w_f > 0)
    between_var = np.where(valid, w_b * w_f * (mu_b - mu_f) ** 2, -1.0)
    thresh = int(np.argmax(between_var))
    return gray_u8 > thresh


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Compute discrete 4-neighbor Laplacian variance without external CV dependencies."""
    arr = gray.astype(np.float64, copy=False)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4.0 * arr[1:-1, 1:-1]
    return float(np.var(lap))


def inspect_image_quality(
    image_path: str | Path,
    cfg: QualityFilterConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a single radiograph against clinical Filter 1 criteria."""
    cfg = cfg or QualityFilterConfig()
    try:
        with Image.open(image_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
    except Exception:
        return {"passed": False, "reason": "empty_or_corrupt", "lap_var": 0.0}

    if gray.size == 0 or int(gray.max()) == int(gray.min()) or int((gray > 5).sum()) < cfg.min_foreground_pixels:
        return {"passed": False, "reason": "empty_or_corrupt", "lap_var": 0.0}

    fg = _otsu_mask(gray)
    if int(fg.sum()) < cfg.min_foreground_pixels:
        return {"passed": False, "reason": "empty_or_corrupt", "lap_var": 0.0}

    h, w = gray.shape
    lap_var = compute_laplacian_variance(gray)
    p01 = float(np.percentile(gray, 0.5))
    p99 = float(np.percentile(gray, 99.5))
    dyn_range = float(p99 - p01)
    overexposed_frac = float((gray >= cfg.overexposed_pixel_thresh).mean())

    gray_f = gray.astype(np.float64)
    gray_norm = np.clip((gray_f - p01) / max(1.0, dyn_range) * 255.0, 0.0, 255.0)
    fg_inner = gray[1:-1, 1:-1] > 15
    if int(fg_inner.sum()) > 50 and h >= 3 and w >= 3:
        lap_norm = (
            gray_norm[:-2, 1:-1]
            + gray_norm[2:, 1:-1]
            + gray_norm[1:-1, :-2]
            + gray_norm[1:-1, 2:]
            - 4.0 * gray_norm[1:-1, 1:-1]
        )
        norm_fg_lap = float(np.var(lap_norm[fg_inner]))
    else:
        norm_fg_lap = 0.0

    ctr = gray[int(0.30 * h) : int(0.70 * h), int(0.30 * w) : int(0.70 * w)]
    center_mean = float(ctr.mean()) if ctr.size > 0 else 0.0
    center_fg_frac = float((ctr > 15).mean()) if ctr.size > 0 else 0.0

    ys, xs = np.where(fg)
    cy, cx = float(ys.mean()), float(xs.mean())
    dx = abs(cx - w / 2.0)
    dy = abs(cy - h / 2.0)
    dist = float(np.hypot(dx, dy))

    area_frac = float(fg.mean())
    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    aspect_ratio = bw / max(1.0, bh)

    pts = np.column_stack([xs - cx, ys - cy])
    if len(pts) >= 2:
        cov = np.cov(pts, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        vec = evecs[:, int(np.argmax(evals))]
        tilt_deg = float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]))))
    else:
        tilt_deg = 0.0

    # Assign primary failure category in priority order
    reason = "kept"
    if (
        lap_var < cfg.min_laplacian_var
        or norm_fg_lap < cfg.min_norm_fg_laplacian_var
        or dyn_range < cfg.min_dynamic_range
        or overexposed_frac > cfg.max_overexposed_frac
        or center_mean < cfg.min_center_mean
    ):
        reason = "blurred_or_obscured"
    elif (
        center_fg_frac < cfg.min_center_fg_frac
        or dx > cfg.max_center_dx
        or dy > cfg.max_center_dy
        or dist > cfg.max_center_dist
    ):
        reason = "misaligned_center"
    elif (
        aspect_ratio < cfg.min_aspect_ratio
        or aspect_ratio > cfg.max_aspect_ratio
        or area_frac < cfg.min_bone_area_frac
        or area_frac > cfg.max_bone_area_frac
    ):
        reason = "distorted_or_stretched"
    elif tilt_deg > cfg.max_tilt_deg:
        reason = "tilted_axis"

    return {
        "passed": reason == "kept",
        "reason": reason,
        "lap_var": lap_var,
        "norm_fg_lap": norm_fg_lap,
        "dyn_range": dyn_range,
        "overexposed_frac": overexposed_frac,
        "center_mean": center_mean,
        "center_fg_frac": center_fg_frac,
        "center_dx": dx,
        "center_dy": dy,
        "center_dist": dist,
        "aspect_ratio": aspect_ratio,
        "bone_area_frac": area_frac,
        "tilt_deg": tilt_deg,
    }


def filter_manifest_by_quality(
    manifest_df: pd.DataFrame,
    images_dir: str | Path,
    cfg: QualityFilterConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Filter raw metadata rows using Filter 1 and return (kept_df, audit_df, report)."""
    cfg = cfg or QualityFilterConfig()
    img_root = Path(images_dir)

    audit_rows: list[dict[str, Any]] = []
    keep_mask: list[bool] = []
    for row in manifest_df.to_dict("records"):
        res = inspect_image_quality(img_root / str(row["filename"]), cfg=cfg)
        keep_mask.append(bool(res["passed"]))
        audit_rows.append(
            {
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "joint": row["joint"],
                "kl_grade": int(row["kl_grade"]),
                "filename": row["filename"],
                **res,
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    kept_df = manifest_df.loc[keep_mask].reset_index(drop=True)
    rejected_df = audit_df[~audit_df["passed"]]

    reason_counts = {str(k): int(v) for k, v in rejected_df["reason"].value_counts().items()}
    kl_before = {str(int(k)): int(v) for k, v in manifest_df["kl_grade"].value_counts().sort_index().items()}
    kl_after = {str(int(k)): int(v) for k, v in kept_df["kl_grade"].value_counts().sort_index().items()}

    report = {
        "config": asdict(cfg),
        "total_input": int(len(manifest_df)),
        "total_kept": int(len(kept_df)),
        "total_rejected": int(len(rejected_df)),
        "rejection_rate_pct": round(100.0 * len(rejected_df) / max(1, len(manifest_df)), 2),
        "rejected_by_reason": reason_counts,
        "kl_before": kl_before,
        "kl_after": kl_after,
    }
    return kept_df, audit_df, report
