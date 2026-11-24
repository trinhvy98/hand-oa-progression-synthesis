"""Val-only logit-bias calibration evaluation for a saved classifier checkpoint."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hand_oa.data import (  # noqa: E402
    JointClassificationDataset,
    build_classifier_transform,
    load_manifest_from_config,
)
from hand_oa.evaluation import build_classifier, compute_classification_metrics  # noqa: E402
from hand_oa.utils import ensure_dir, load_yaml_config, resolve_project_path, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate class logit biases strictly on validation logits and evaluate on locked test."
    )
    parser.add_argument("--data-config", default="configs/data_filtered_v2.yaml")
    parser.add_argument("--config", default="configs/classifier_filtered.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="vgg19")
    parser.add_argument(
        "--hierarchical-oa",
        action="store_true",
        help="Promote <3 predictions when P(OA)=P(KL3)+P(KL4) > 0.50.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward pass once and return (logits_np, targets_np)."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for batch in loader:
        images, targets = batch[0], batch[1]
        logits = model(images.to(device, non_blocking=True))
        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def compute_metrics_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    bias: np.ndarray | list[float] | None = None,
    hierarchical_oa: bool = False,
) -> dict[str, Any]:
    """Compute exact classification + per-class precision/recall + KL4-vs-KL3 separability metrics."""
    b = np.zeros(logits.shape[1], dtype=np.float64) if bias is None else np.asarray(bias, dtype=np.float64)
    shifted = logits.astype(np.float64) + b[np.newaxis, :]
    shifted -= shifted.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    if hierarchical_oa:
        oa_prob = probs[:, 3:].sum(axis=1)
        promote_mask = (preds < 3) & (oa_prob > 0.50)
        if np.any(promote_mask):
            preds = preds.copy()
            preds[promote_mask] = 3 + np.argmax(probs[promote_mask, 3:5], axis=1)

    base = compute_classification_metrics(
        y_true=targets,
        y_pred=preds,
        y_prob=probs,
        num_classes=5,
        oa_threshold_kl=3,
    )
    prec, rec, _, _ = precision_recall_fscore_support(
        targets, preds, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    severe_mask = targets >= 3
    if severe_mask.sum() > 0 and len(np.unique(targets[severe_mask])) == 2:
        p_sev = probs[severe_mask]
        kl4_cond_prob = p_sev[:, 4] / np.maximum(1e-12, p_sev[:, 3] + p_sev[:, 4])
        severe_auroc = float(roc_auc_score((targets[severe_mask] == 4).astype(int), kl4_cond_prob))
    else:
        severe_auroc = 0.0

    return {
        **base,
        "per_class_recall": {f"KL{i}": float(rec[i]) for i in range(5)},
        "per_class_precision": {f"KL{i}": float(prec[i]) for i in range(5)},
        "severe_kl4_vs_kl3_auroc": severe_auroc,
    }


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml_config(resolve_project_path(args.data_config, _ROOT))
    cls_cfg = load_yaml_config(resolve_project_path(args.config, _ROOT))
    manifest, images_dir = load_manifest_from_config(data_cfg, _ROOT, overwrite=False)

    joints = list(data_cfg["dataset"]["joints"])
    image_size = int(cls_cfg["data"]["image_size"])
    tf_eval = build_classifier_transform(image_size=image_size, is_train=False)

    val_ds = JointClassificationDataset(
        manifest_df=manifest,
        images_dir=images_dir,
        split="val",
        joints=joints,
        transform=tf_eval,
    )
    test_ds = JointClassificationDataset(
        manifest_df=manifest,
        images_dir=images_dir,
        split="test",
        joints=joints,
        transform=tf_eval,
    )
    pin = torch.cuda.is_available()
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(
        backbone=args.backbone,
        num_classes=int(cls_cfg["model"]["num_classes"]),
        pretrained=False,
        dropout=float(cls_cfg["model"]["dropout"]),
    )
    ckpt_path = resolve_project_path(args.checkpoint, _ROOT)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    val_logits, val_targets = collect_logits(model, val_loader, device)
    test_logits, test_targets = collect_logits(model, test_loader, device)
    hier = bool(args.hierarchical_oa)

    # Strict val-only search (never inspects test logits)
    best_a: tuple[float, float, list[float]] | None = None
    for b4 in np.arange(-0.5, 3.01, 0.05):
        bias = [0.0, 0.0, 0.0, 0.0, float(round(b4, 4))]
        m = compute_metrics_from_logits(val_logits, val_targets, bias, hierarchical_oa=hier)
        score = 0.5 * (m["per_class_f1"]["KL3"] + m["per_class_f1"]["KL4"])
        key = (score, m["macro_f1"])
        if best_a is None or key > (best_a[0], best_a[1]):
            best_a = (score, m["macro_f1"], bias)

    best_b1: tuple[float, float, list[float]] | None = None
    best_b2: tuple[float, float, list[float]] | None = None
    for b3 in np.arange(-0.5, 2.51, 0.10):
        for b4 in np.arange(-0.5, 3.01, 0.10):
            bias = [0.0, 0.0, 0.0, float(round(b3, 4)), float(round(b4, 4))]
            m = compute_metrics_from_logits(val_logits, val_targets, bias, hierarchical_oa=hier)
            f1_key = (0.5 * (m["per_class_f1"]["KL3"] + m["per_class_f1"]["KL4"]), m["macro_f1"])
            if best_b1 is None or f1_key > (best_b1[0], best_b1[1]):
                best_b1 = (f1_key[0], f1_key[1], bias)
            if m["accuracy"] >= 0.75:
                sens_score = (
                    0.35 * m["per_class_f1"]["KL3"]
                    + 0.35 * m["per_class_f1"]["KL4"]
                    + 0.30 * m["binary_oa_sensitivity"]
                )
                if best_b2 is None or (sens_score, m["macro_f1"]) > (best_b2[0], best_b2[1]):
                    best_b2 = (sens_score, m["macro_f1"], bias)

    assert best_a is not None and best_b1 is not None
    methods: dict[str, list[float]] = {
        "uncalibrated": [0.0, 0.0, 0.0, 0.0, 0.0],
        "val_b4_only": best_a[2],
        "val_b3_b4_f1": best_b1[2],
    }
    if best_b2 is not None:
        methods["val_b3_b4_high_sens"] = best_b2[2]

    results: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "backbone": args.backbone,
        "hierarchical_oa": hier,
        "val_samples": int(len(val_targets)),
        "test_samples": int(len(test_targets)),
        "methods": {},
    }

    print("Method | Bias[3,4] | Acc(%) | MacroF1 | KL3_F1(Rec/Pre) | KL4_F1(Rec/Pre) | BinSens | BinAUC | KL4v3AUC")
    for name, bias in methods.items():
        val_m = compute_metrics_from_logits(val_logits, val_targets, bias, hierarchical_oa=hier)
        test_m = compute_metrics_from_logits(test_logits, test_targets, bias, hierarchical_oa=hier)
        results["methods"][name] = {
            "bias": bias,
            "val_metrics": val_m,
            "test_metrics": test_m,
        }
        print(
            f"{name:20s} | [{bias[3]:+.2f},{bias[4]:+.2f}] | "
            f"{100.0 * test_m['accuracy']:5.2f}% | {test_m['macro_f1']:.4f} | "
            f"{test_m['per_class_f1']['KL3']:.4f} ({test_m['per_class_recall']['KL3']:.2f}/{test_m['per_class_precision']['KL3']:.2f}) | "
            f"{test_m['per_class_f1']['KL4']:.4f} ({test_m['per_class_recall']['KL4']:.2f}/{test_m['per_class_precision']['KL4']:.2f}) | "
            f"{100.0 * test_m['binary_oa_sensitivity']:5.2f}% | {test_m['binary_oa_roc_auc']:.4f} | "
            f"{test_m['severe_kl4_vs_kl3_auroc']:.4f}"
        )

    out_path = resolve_project_path(args.output_json, _ROOT)
    ensure_dir(out_path.parent)
    save_json(results, out_path)
    print(f"Saved calibrated metrics JSON to: {out_path}")


if __name__ == "__main__":
    main()
