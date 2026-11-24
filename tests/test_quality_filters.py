"""Unit tests for clinical Filter 1 and expert Filter 2 metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cyclegan.scripts.filter_synthetic_kl3 import (
    compute_pair_artifact_metrics,
    evaluate_filter2_candidate,
    evaluate_filter2_physical_only,
)
from hand_oa.data import (
    QualityFilterConfig,
    filter_manifest_by_quality,
    inspect_image_quality,
)


def _make_joint_image(
    size: int = 180,
    dx: int = 0,
    dy: int = 0,
    tilt_deg: float = 0.0,
    blur: bool = False,
    intensity_scale: float = 1.0,
) -> np.ndarray:
    """Create a synthetic radiograph-like vertical phalanx joint patch."""
    arr = np.zeros((size, size), dtype=np.uint8)
    cx = size // 2 + dx
    cy = size // 2 + dy
    rad = np.deg2rad(tilt_deg)
    yy, xx = np.mgrid[0:size, 0:size]
    rx = (xx - cx) * np.cos(rad) + (yy - cy) * np.sin(rad)
    ry = -(xx - cx) * np.sin(rad) + (yy - cy) * np.cos(rad)

    bone = (np.abs(rx) < 22) & (np.abs(ry) < 70) & (np.abs(ry) > 4)
    arr[bone] = int(round(185 * intensity_scale))
    # Add realistic trabecular texture so Laplacian variance is high
    rng = np.random.default_rng(7)
    tex = rng.integers(-25, 26, size=arr.shape)
    arr = np.where(bone, np.clip(arr.astype(int) + np.round(tex * intensity_scale).astype(int), 25, 235).astype(np.uint8), arr)
    if blur:
        pil = Image.fromarray(arr).resize((30, 30), Image.Resampling.BILINEAR).resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(pil, dtype=np.uint8)
    return arr


def test_filter1_rejects_corrupt_blur_offcenter_and_keeps_mild_oa_tilt(tmp_path: Path) -> None:
    cfg = QualityFilterConfig()
    samples = {
        "clean_kl0.png": _make_joint_image(dx=0, dy=0, tilt_deg=4.0),
        "oa_tilted_26deg.png": _make_joint_image(dx=4, dy=-4, tilt_deg=26.0),
        "dark_sharp.png": _make_joint_image(dx=0, dy=0, tilt_deg=2.0, intensity_scale=0.45),
        "extreme_tilt_55deg.png": _make_joint_image(dx=0, dy=0, tilt_deg=55.0),
        "offcenter_45px.png": _make_joint_image(dx=45, dy=0, tilt_deg=0.0),
        "blurry.png": _make_joint_image(blur=True),
        "empty.png": np.zeros((180, 180), dtype=np.uint8),
    }
    rows = []
    for idx, (fname, arr) in enumerate(samples.items()):
        Image.fromarray(arr, mode="L").save(tmp_path / fname)
        rows.append(
            {
                "sample_id": f"p{idx}_dip2",
                "patient_id": f"p{idx}",
                "joint": "DIP2",
                "kl_grade": 3 if "oa" in fname else 0,
                "filename": fname,
            }
        )

    assert inspect_image_quality(tmp_path / "clean_kl0.png", cfg)["passed"] is True
    assert inspect_image_quality(tmp_path / "oa_tilted_26deg.png", cfg)["passed"] is True
    assert inspect_image_quality(tmp_path / "dark_sharp.png", cfg)["passed"] is True
    assert inspect_image_quality(tmp_path / "extreme_tilt_55deg.png", cfg)["reason"] == "tilted_axis"
    assert inspect_image_quality(tmp_path / "offcenter_45px.png", cfg)["reason"] == "misaligned_center"
    assert inspect_image_quality(tmp_path / "blurry.png", cfg)["reason"] == "blurred_or_obscured"
    assert inspect_image_quality(tmp_path / "empty.png", cfg)["reason"] == "empty_or_corrupt"

    kept_df, _, report = filter_manifest_by_quality(pd.DataFrame(rows), tmp_path, cfg)
    assert set(kept_df["filename"]) == {"clean_kl0.png", "oa_tilted_26deg.png", "dark_sharp.png"}
    assert report["total_rejected"] == 4


def test_filter2_pair_metrics_detect_bg_ghosting_and_joint_focus() -> None:
    src = _make_joint_image(size=256)
    good_syn = src.copy()
    # Edit focused inside middle articular band (rows 95..160) within bone
    good_syn[105:150, 108:148] = np.clip(good_syn[105:150, 108:148].astype(int) + 35, 0, 255).astype(np.uint8)
    m_good = compute_pair_artifact_metrics(src, good_syn)
    assert m_good["bg_diff"] == 0.0
    assert m_good["joint_focus_ratio"] > 1.5

    ghost_syn = src.copy()
    ghost_syn[src < 15] = 40
    m_ghost = compute_pair_artifact_metrics(src, ghost_syn)
    assert m_ghost["bg_diff"] > 20.0


def test_filter2_pure_kl3_gate_blocks_kl4_overshoot() -> None:
    good_phys = {
        "bg_diff": 0.4,
        "joint_focus_ratio": 0.85,
        "lap_ratio": 1.1,
        "struct_ssim": 0.78,
        "delta_l1": 0.08,
    }
    probs_pure_kl3 = np.array([0.02, 0.03, 0.12, 0.70, 0.13], dtype=np.float32)
    phys_ok, passed, reason = evaluate_filter2_candidate(good_phys, probs_pure_kl3)
    assert phys_ok and passed and reason == "passed"

    probs_kl4_overshoot = np.array([0.01, 0.02, 0.05, 0.38, 0.54], dtype=np.float32)
    phys_ok, passed, reason = evaluate_filter2_candidate(good_phys, probs_kl4_overshoot)
    assert phys_ok and (not passed) and reason == "oracle_kl4_overshoot"

    probs_borderline_kl3 = np.array([0.05, 0.05, 0.48, 0.40, 0.02], dtype=np.float32)
    phys_ok, passed, reason = evaluate_filter2_candidate(good_phys, probs_borderline_kl3)
    assert phys_ok and passed and reason == "passed"

    probs_low_oa = np.array([0.05, 0.05, 0.70, 0.18, 0.02], dtype=np.float32)
    phys_ok, passed, reason = evaluate_filter2_candidate(good_phys, probs_low_oa)
    assert phys_ok and (not passed) and reason == "oracle_insufficient_oa"

    probs_kl2_argmax = np.array([0.02, 0.02, 0.49, 0.46, 0.01], dtype=np.float32)
    phys_ok, passed, reason = evaluate_filter2_candidate(
        good_phys, probs_kl2_argmax, min_kl3_prob=0.45, require_kl3_argmax=True
    )
    assert phys_ok and (not passed) and reason == "oracle_not_kl3_argmax"

    probs_pure_kl4 = np.array([0.01, 0.02, 0.04, 0.18, 0.75], dtype=np.float32)
    phys_ok, passed, reason = evaluate_filter2_candidate(good_phys, probs_pure_kl4, target_kl=4)
    assert phys_ok and passed and reason == "passed"
    phys_ok, passed, reason = evaluate_filter2_candidate(good_phys, probs_pure_kl3, target_kl=4)
    assert phys_ok and (not passed) and reason == "oracle_kl3_bleed"


def test_filter2_physical_only_requires_no_oracle() -> None:
    good_phys = {
        "bg_diff": 0.8,
        "joint_focus_ratio": 0.90,
        "lap_ratio": 1.2,
        "struct_ssim": 0.82,
        "delta_l1": 0.08,
    }
    assert evaluate_filter2_physical_only(good_phys) == (True, True, "passed")
    assert evaluate_filter2_physical_only({**good_phys, "bg_diff": 4.0}) == (False, False, "bg_ghosting")
    assert evaluate_filter2_physical_only({**good_phys, "struct_ssim": 0.50}) == (False, False, "anatomy_ssim")
    assert evaluate_filter2_physical_only({**good_phys, "struct_ssim": 0.98}) == (False, False, "anatomy_ssim")
    assert evaluate_filter2_physical_only({**good_phys, "delta_l1": 0.01}) == (False, False, "delta_l1_bounds")
    assert evaluate_filter2_physical_only({**good_phys, "lap_ratio": 3.5}) == (False, False, "texture_laplacian")
    assert evaluate_filter2_physical_only({**good_phys, "joint_focus_ratio": 0.4}) == (False, False, "off_joint_edit")
