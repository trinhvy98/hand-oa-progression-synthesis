"""Expert automated Filter 2 for screening CycleGAN synthetic KL3 radiographs."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hand_oa.data import (  # noqa: E402
    build_classifier_transform,
    compute_laplacian_variance,
    load_manifest_from_config,
)
from hand_oa.evaluation import build_classifier, compute_batch_ssim  # noqa: E402
from hand_oa.utils import ensure_dir, load_yaml_config, resolve_project_path, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply expert Filter 2 (artifact + clinical OA check) on synthetic KL3 images."
    )
    parser.add_argument("--data-config", default="configs/data_filtered.yaml")
    parser.add_argument("--cyclegan-config", default="configs/cyclegan_filtered.yaml")
    parser.add_argument("--classifier-config", default="configs/classifier_filtered.yaml")
    parser.add_argument(
        "--synthetic-manifest",
        default="outputs/cyclegan_filtered/synthetic_kl3/synthetic_manifest.csv",
    )
    parser.add_argument(
        "--oracle-checkpoint",
        default=None,
        help="Optional path to classifier checkpoint for Oracle screening; omit or pass --physical-only for 0% AI filtering.",
    )
    parser.add_argument(
        "--physical-only",
        action="store_true",
        help="Filter strictly using physical/anatomical metrics with zero AI/Oracle bias.",
    )
    parser.add_argument(
        "--oracle-backbone",
        default=None,
        help="Optional backbone override for the Oracle checkpoint (e.g. densenet121).",
    )
    parser.add_argument(
        "--output-manifest",
        default="outputs/cyclegan_filtered/synthetic_kl3/synthetic_manifest_filtered.csv",
    )
    parser.add_argument("--max-bg-diff", type=float, default=2.5)
    parser.add_argument("--min-ssim", type=float, default=0.60)
    parser.add_argument("--max-ssim", type=float, default=0.95)
    parser.add_argument("--min-delta-l1", type=float, default=0.035)
    parser.add_argument("--max-delta-l1", type=float, default=0.24)
    parser.add_argument("--min-lap-ratio", type=float, default=0.60)
    parser.add_argument("--max-lap-ratio", type=float, default=2.80)
    parser.add_argument("--min-joint-focus", type=float, default=0.70)
    parser.add_argument("--min-oa-prob", type=float, default=0.30)
    parser.add_argument("--max-kl0-prob", type=float, default=0.20)
    parser.add_argument("--min-kl3-prob", type=float, default=0.25)
    parser.add_argument("--max-kl4-prob", type=float, default=0.30)
    parser.add_argument("--min-kl3-margin", type=float, default=0.05)
    parser.add_argument(
        "--target-kl",
        type=int,
        default=3,
        choices=[3, 4],
        help="Target OA severity grade to verify (3 or 4).",
    )
    parser.add_argument(
        "--require-kl3-argmax",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require Oracle argmax prediction to match --target-kl cleanly.",
    )
    parser.add_argument("--min-keep", type=int, default=40)
    parser.add_argument(
        "--max-per-patient",
        type=int,
        default=2,
        help="Maximum accepted synthetic joints per source patient (0 disables cap).",
    )
    parser.add_argument(
        "--max-accepted",
        type=int,
        default=None,
        help="Optional random diversity cap on accepted synthetic samples (uses seed=42).",
    )
    return parser.parse_args()


def evaluate_filter2_physical_only(
    metrics: dict[str, float],
    *,
    max_bg_diff: float = 2.5,
    min_ssim: float = 0.60,
    max_ssim: float = 0.95,
    min_delta_l1: float = 0.035,
    max_delta_l1: float = 0.24,
    min_lap_ratio: float = 0.60,
    max_lap_ratio: float = 2.80,
    min_joint_focus: float = 0.70,
) -> tuple[bool, bool, str]:
    """Screen a synthetic radiograph strictly by physical/anatomical artifact gates (zero AI)."""
    if metrics["bg_diff"] > max_bg_diff:
        return False, False, "bg_ghosting"
    if not (min_ssim <= metrics["struct_ssim"] <= max_ssim):
        return False, False, "anatomy_ssim"
    if not (min_delta_l1 <= metrics["delta_l1"] <= max_delta_l1):
        return False, False, "delta_l1_bounds"
    if not (min_lap_ratio <= metrics["lap_ratio"] <= max_lap_ratio):
        return False, False, "texture_laplacian"
    if metrics["joint_focus_ratio"] < min_joint_focus:
        return False, False, "off_joint_edit"
    return True, True, "passed"


def evaluate_filter2_candidate(
    metrics: dict[str, float],
    probs_row: np.ndarray,
    *,
    max_bg_diff: float = 2.5,
    min_ssim: float = 0.60,
    max_ssim: float = 0.95,
    min_delta_l1: float = 0.035,
    max_delta_l1: float = 0.24,
    min_lap_ratio: float = 0.60,
    max_lap_ratio: float = 2.80,
    min_joint_focus: float = 0.70,
    min_oa_prob: float = 0.30,
    max_kl0_prob: float = 0.20,
    min_kl3_prob: float = 0.25,
    max_kl4_prob: float = 0.30,
    min_kl3_margin: float = 0.05,
    require_kl3_argmax: bool = False,
    target_kl: int = 3,
) -> tuple[bool, bool, str]:
    """Apply physical artifact gates followed by pure-target (KL3 or KL4) Oracle clinical checks."""
    phys_ok, _, phys_reason = evaluate_filter2_physical_only(
        metrics,
        max_bg_diff=max_bg_diff,
        min_ssim=min_ssim,
        max_ssim=max_ssim,
        min_delta_l1=min_delta_l1,
        max_delta_l1=max_delta_l1,
        min_lap_ratio=min_lap_ratio,
        max_lap_ratio=max_lap_ratio,
        min_joint_focus=min_joint_focus,
    )
    if not phys_ok:
        return False, False, phys_reason

    p_kl0 = float(probs_row[0])
    p_kl3 = float(probs_row[3])
    p_kl4 = float(probs_row[4])
    prob_oa = p_kl3 + p_kl4
    pred_kl = int(np.argmax(probs_row))
    target_idx = int(target_kl)
    p_target = p_kl3 if target_idx == 3 else p_kl4
    p_other_oa = p_kl4 if target_idx == 3 else p_kl3

    if prob_oa < min_oa_prob or p_kl0 > max_kl0_prob:
        return True, False, "oracle_insufficient_oa"
    if target_idx == 3:
        if p_other_oa > max_kl4_prob or (p_target - p_other_oa) < min_kl3_margin:
            return True, False, "oracle_kl4_overshoot"
        if p_target < min_kl3_prob:
            return True, False, "oracle_low_kl3_prob"
        if require_kl3_argmax and pred_kl != 3:
            return True, False, "oracle_not_kl3_argmax"
    else:
        if (p_target - p_other_oa) < min_kl3_margin:
            return True, False, "oracle_kl3_bleed"
        if p_target < min_kl3_prob:
            return True, False, "oracle_low_kl4_prob"
        if require_kl3_argmax and pred_kl != 4:
            return True, False, "oracle_not_kl4_argmax"
    return True, True, "passed"


def compute_pair_artifact_metrics(src_u8: np.ndarray, gen_u8: np.ndarray) -> dict[str, float]:
    """Measure background ghosting, structural preservation, texture sharpness, and joint focus."""
    src_f = src_u8.astype(np.float32)
    gen_f = gen_u8.astype(np.float32)
    diff = np.abs(gen_f - src_f)

    h, w = src_u8.shape
    articular_box = np.zeros((h, w), dtype=bool)
    articular_box[int(0.35 * h) : int(0.65 * h), int(0.30 * w) : int(0.70 * w)] = True
    bg_mask = (src_u8 < 15) & (~articular_box)
    bg_diff = float(diff[bg_mask].mean()) if bg_mask.any() else 0.0

    # Articular zone (middle 40% height) vs distal/proximal outer bone shaft
    fg = src_u8 >= 15
    mid_band = np.zeros_like(fg)
    mid_band[int(0.30 * h) : int(0.70 * h), :] = True
    outer_band = np.zeros_like(fg)
    outer_band[: int(0.20 * h), :] = True
    outer_band[int(0.80 * h) :, :] = True

    mid_diff = float(diff[fg & mid_band].mean()) if (fg & mid_band).any() else 0.0
    out_diff = float(diff[fg & outer_band].mean()) if (fg & outer_band).any() else 1e-6
    joint_focus_ratio = mid_diff / max(1e-6, out_diff)

    lap_src = compute_laplacian_variance(src_u8)
    lap_gen = compute_laplacian_variance(gen_u8)
    lap_ratio = lap_gen / max(1.0, lap_src)

    src_t = torch.from_numpy(src_f / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)
    gen_t = torch.from_numpy(gen_f / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)
    struct_ssim = compute_batch_ssim(src_t, gen_t)
    delta_l1 = float(torch.abs(gen_t - src_t).mean().item())

    return {
        "bg_diff": bg_diff,
        "joint_focus_ratio": joint_focus_ratio,
        "lap_ratio": lap_ratio,
        "struct_ssim": struct_ssim,
        "delta_l1": delta_l1,
    }


@torch.no_grad()
def score_with_oracle(
    image_paths: list[Path],
    cls_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    backbone_override: str | None = None,
) -> np.ndarray:
    """Predict KL0-KL4 class probabilities using a real-data Oracle classifier."""
    model_cfg = cls_cfg["model"]
    backbone = backbone_override or str(model_cfg["backbone"])
    model = build_classifier(
        backbone=backbone,
        num_classes=int(model_cfg["num_classes"]),
        pretrained=False,
        dropout=float(model_cfg["dropout"]),
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    tf = build_classifier_transform(image_size=int(cls_cfg["data"]["image_size"]), is_train=False)
    probs_list: list[np.ndarray] = []
    batch_tensors: list[torch.Tensor] = []

    for p in image_paths:
        with Image.open(p) as img:
            batch_tensors.append(tf(img.copy()))
        if len(batch_tensors) == 32:
            x = torch.stack(batch_tensors, dim=0).to(device)
            probs_list.append(torch.softmax(model(x), dim=1).cpu().numpy())
            batch_tensors.clear()
    if batch_tensors:
        x = torch.stack(batch_tensors, dim=0).to(device)
        probs_list.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(probs_list, axis=0)


def _save_preview_grid(audit_df: pd.DataFrame, out_path: Path, physical_only: bool = False) -> None:
    sort_col = "joint_focus_ratio" if physical_only else "prob_kl3"
    acc = audit_df[audit_df["passed"]].sort_values(sort_col, ascending=False).head(4)
    rej = audit_df[~audit_df["passed"]].head(4)
    if acc.empty and rej.empty:
        return

    ncols = max(len(acc), len(rej), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.4), dpi=130)
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (sub, label_prefix, color) in enumerate(
        [(acc, "ACCEPTED", "#059669"), (rej, "REJECTED", "#DC2626")]
    ):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= len(sub):
                continue
            r = sub.iloc[col_idx]
            with Image.open(r["image_path"]) as im:
                ax.imshow(im.convert("L"), cmap="gray", vmin=0, vmax=255)
            metric_line = (
                f"SSIM={r['struct_ssim']:.2f} JFR={r['joint_focus_ratio']:.2f} BG={r['bg_diff']:.2f}"
                if physical_only
                else f"OA={r['prob_oa']:.2f} SSIM={r['struct_ssim']:.2f} BG={r['bg_diff']:.2f}"
            )
            ax.set_title(
                f"{label_prefix} ({r['reject_reason']})\n{metric_line}",
                fontsize=7.8,
                fontweight="bold",
                color=color,
            )

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml_config(resolve_project_path(args.data_config, _ROOT))
    cg_cfg = load_yaml_config(resolve_project_path(args.cyclegan_config, _ROOT))
    cls_cfg = load_yaml_config(resolve_project_path(args.classifier_config, _ROOT))

    manifest, images_dir = load_manifest_from_config(data_cfg, _ROOT, overwrite=False)
    syn_path = resolve_project_path(args.synthetic_manifest, _ROOT)
    syn_df = pd.read_csv(syn_path)
    if "source_split" in syn_df.columns and (syn_df["source_split"] == "test").any():
        raise AssertionError("Synthetic manifest must never contain samples derived from 'test'!")

    merged = syn_df.merge(
        manifest[["sample_id", "filename", "joint"]],
        left_on="source_sample_id",
        right_on="sample_id",
        how="left",
        suffixes=("", "_src"),
    )

    img_size = int(cg_cfg["data"]["image_size"])
    physical_only = bool(args.physical_only or args.oracle_checkpoint is None)
    if physical_only:
        probs = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oracle_ckpt = resolve_project_path(args.oracle_checkpoint, _ROOT)
        probs = score_with_oracle(
            image_paths=[Path(p) for p in merged["image_path"]],
            cls_cfg=cls_cfg,
            checkpoint_path=oracle_ckpt,
            device=device,
            backbone_override=args.oracle_backbone,
        )

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(merged.to_dict("records")):
        with Image.open(images_dir / str(row["filename"])) as src_im:
            src_u8 = np.asarray(src_im.convert("L").resize((img_size, img_size), Image.Resampling.BICUBIC), dtype=np.uint8)
        with Image.open(row["image_path"]) as gen_im:
            gen_u8 = np.asarray(gen_im.convert("L"), dtype=np.uint8)

        m = compute_pair_artifact_metrics(src_u8, gen_u8)
        if physical_only or probs is None:
            physical_ok, passed, reason = evaluate_filter2_physical_only(
                m,
                max_bg_diff=args.max_bg_diff,
                min_ssim=args.min_ssim,
                max_ssim=args.max_ssim,
                min_delta_l1=args.min_delta_l1,
                max_delta_l1=args.max_delta_l1,
                min_lap_ratio=args.min_lap_ratio,
                max_lap_ratio=args.max_lap_ratio,
                min_joint_focus=args.min_joint_focus,
            )
            prob_fields = {
                "prob_kl0": float("nan"),
                "prob_kl1": float("nan"),
                "prob_kl2": float("nan"),
                "prob_kl3": float("nan"),
                "prob_kl4": float("nan"),
                "prob_oa": float("nan"),
                "kl3_margin": float("nan"),
                "oracle_pred_kl": -1,
            }
        else:
            p_kl0 = float(probs[idx, 0])
            p_kl1 = float(probs[idx, 1])
            p_kl2 = float(probs[idx, 2])
            p_kl3 = float(probs[idx, 3])
            p_kl4 = float(probs[idx, 4])
            prob_oa = p_kl3 + p_kl4
            pred_kl = int(np.argmax(probs[idx]))
            physical_ok, passed, reason = evaluate_filter2_candidate(
                m,
                probs[idx],
                max_bg_diff=args.max_bg_diff,
                min_ssim=args.min_ssim,
                max_ssim=args.max_ssim,
                min_delta_l1=args.min_delta_l1,
                max_delta_l1=args.max_delta_l1,
                min_lap_ratio=args.min_lap_ratio,
                max_lap_ratio=args.max_lap_ratio,
                min_joint_focus=args.min_joint_focus,
                min_oa_prob=args.min_oa_prob,
                max_kl0_prob=args.max_kl0_prob,
                min_kl3_prob=args.min_kl3_prob,
                max_kl4_prob=args.max_kl4_prob,
                min_kl3_margin=args.min_kl3_margin,
                require_kl3_argmax=args.require_kl3_argmax,
                target_kl=args.target_kl,
            )
            target_margin = (p_kl3 - p_kl4) if args.target_kl == 3 else (p_kl4 - p_kl3)
            prob_fields = {
                "prob_kl0": p_kl0,
                "prob_kl1": p_kl1,
                "prob_kl2": p_kl2,
                "prob_kl3": p_kl3,
                "prob_kl4": p_kl4,
                "prob_oa": prob_oa,
                "kl3_margin": target_margin,
                "oracle_pred_kl": pred_kl,
            }

        records.append(
            {
                **row,
                **m,
                **prob_fields,
                "physical_ok": physical_ok,
                "passed": passed,
                "reject_reason": reason,
            }
        )

    audit_df = pd.DataFrame(records)
    kept_df = audit_df[audit_df["passed"]].copy()

    used_fallback = False
    if len(kept_df) < args.min_keep:
        if physical_only:
            relaxed = audit_df[
                (audit_df["bg_diff"] <= args.max_bg_diff * 1.5)
                & (audit_df["struct_ssim"] >= args.min_ssim - 0.05)
                & (audit_df["struct_ssim"] <= args.max_ssim + 0.02)
            ].sort_values("joint_focus_ratio", ascending=False)
            if len(relaxed) > len(kept_df):
                target_n = min(max(args.min_keep, len(kept_df)), len(relaxed))
                top_ids = set(relaxed.head(target_n)["sample_id"])
                new_only = audit_df["sample_id"].isin(top_ids) & (~audit_df["passed"])
                audit_df.loc[new_only, "passed"] = True
                audit_df.loc[new_only, "reject_reason"] = "passed_relaxed_physical"
                kept_df = audit_df[audit_df["passed"]].copy()
                used_fallback = True
        else:
            if args.target_kl == 3:
                purity_mask = (audit_df["prob_kl3"] > audit_df["prob_kl4"]) & (
                    audit_df["prob_kl4"] <= args.max_kl4_prob
                )
            else:
                purity_mask = (audit_df["prob_kl4"] > audit_df["prob_kl3"]) & (
                    audit_df["oracle_pred_kl"] == 4
                )
            phys_only = audit_df[
                audit_df["physical_ok"] & (audit_df["prob_kl0"] < 0.35) & purity_mask
            ].sort_values("kl3_margin", ascending=False)
            if len(phys_only) > len(kept_df):
                target_n = min(max(args.min_keep, len(kept_df)), len(phys_only))
                top_ids = set(phys_only.head(target_n)["sample_id"])
                new_only = audit_df["sample_id"].isin(top_ids) & (~audit_df["passed"])
                audit_df.loc[new_only, "passed"] = True
                audit_df.loc[new_only, "reject_reason"] = "passed_relaxed_oracle"
                kept_df = audit_df[audit_df["passed"]].copy()
                used_fallback = True

    if args.max_per_patient > 0 and "source_patient_id" in kept_df.columns and not kept_df.empty:
        ordered = (
            kept_df.sample(frac=1.0, random_state=42)
            if physical_only
            else kept_df.sort_values("kl3_margin", ascending=False)
        )
        kept_df = ordered.groupby("source_patient_id", sort=False).head(int(args.max_per_patient)).reset_index(drop=True)

    if args.max_accepted is not None and len(kept_df) > int(args.max_accepted):
        kept_df = kept_df.sample(n=int(args.max_accepted), random_state=42).reset_index(drop=True)

    out_manifest = resolve_project_path(args.output_manifest, _ROOT)
    ensure_dir(out_manifest.parent)
    syn_cols = list(syn_df.columns)
    kept_df[syn_cols].to_csv(out_manifest, index=False)
    audit_df.to_csv(out_manifest.with_name("filter2_audit.csv"), index=False)
    _save_preview_grid(audit_df, out_manifest.with_name("filter2_accepted_vs_rejected.png"), physical_only=physical_only)

    report = {
        "mode": "physical_only" if physical_only else "oracle_assisted",
        "total_candidates": int(len(audit_df)),
        "accepted_count": int(len(kept_df)),
        "accepted_unique_patients": int(kept_df["source_patient_id"].nunique())
        if (not kept_df.empty and "source_patient_id" in kept_df.columns)
        else 0,
        "acceptance_rate_pct": round(100.0 * len(kept_df) / max(1, len(audit_df)), 2),
        "used_oracle_fallback": used_fallback,
        "reason_breakdown": {str(k): int(v) for k, v in audit_df["reject_reason"].value_counts().items()},
        "accepted_oracle_pred_kl": {
            str(k): int(v) for k, v in kept_df["oracle_pred_kl"].value_counts().sort_index().items()
        }
        if (not physical_only and not kept_df.empty)
        else {},
        "accepted_kl4_gt_kl3_count": int((kept_df["prob_kl4"] > kept_df["prob_kl3"]).sum())
        if (not physical_only and not kept_df.empty)
        else 0,
        "accepted_mean_prob_kl3": round(float(kept_df["prob_kl3"].mean()), 4) if (not physical_only and not kept_df.empty) else None,
        "accepted_mean_prob_kl4": round(float(kept_df["prob_kl4"].mean()), 4) if (not physical_only and not kept_df.empty) else None,
        "accepted_mean_prob_oa": round(float(kept_df["prob_oa"].mean()), 4) if (not physical_only and not kept_df.empty) else None,
        "accepted_mean_ssim": round(float(kept_df["struct_ssim"].mean()), 4) if not kept_df.empty else 0.0,
        "accepted_mean_joint_focus_ratio": round(float(kept_df["joint_focus_ratio"].mean()), 4) if not kept_df.empty else 0.0,
    }
    save_json(report, out_manifest.with_name("filter2_report.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
