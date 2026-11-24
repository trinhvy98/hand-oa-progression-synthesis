"""Shared downstream evaluation script comparing baseline vs synthetic augmentation."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from hand_oa.evaluation import (  # noqa: E402
    build_classifier,
    build_fast_classification_loaders,
    evaluate_classifier,
    train_classifier,
)
from hand_oa.utils import (  # noqa: E402
    ensure_dir,
    load_yaml_config,
    resolve_project_path,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate downstream KL classifier on locked test set."
    )
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to data YAML.")
    parser.add_argument(
        "--config", default="configs/classifier.yaml", help="Path to classifier YAML."
    )
    parser.add_argument(
        "--synthetic-manifest",
        default=None,
        help="Optional synthetic_manifest.csv produced by CycleGAN or Diffusion.",
    )
    parser.add_argument("--max-synthetic", type=int, default=None, help="Cap synthetic samples.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument(
        "--loss-type",
        default=None,
        choices=["ce", "focal"],
        help="Override loss function ('ce' or 'focal').",
    )
    parser.add_argument("--oversample-kl3", type=int, default=None, help="Oversample multiplier for KL3.")
    parser.add_argument("--oversample-kl4", type=int, default=None, help="Oversample multiplier for KL4.")
    parser.add_argument(
        "--synthetic-repeat-kl3",
        type=int,
        default=None,
        help="Optional separate repeat multiplier for synthetic KL3 items (prevents synthetic domain domination).",
    )
    parser.add_argument(
        "--synthetic-repeat-kl4",
        type=int,
        default=None,
        help="Optional separate repeat multiplier for synthetic KL4 items.",
    )
    parser.add_argument(
        "--save-checkpoint",
        default=None,
        help="Optional file path to save best trained classifier weights.",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip test-set evaluation and metric file creation (e.g. when training internal Filter-2 Oracle).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory overriding classifier experiment.output_dir.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seeds for multi-seed replication and mean/std summary.",
    )
    parser.add_argument(
        "--focal-config-json",
        default=None,
        help="Optional JSON file produced by tune_focal_hyperopt.py.",
    )
    parser.add_argument("--focal-gamma", type=float, default=None, help="Override focal gamma.")
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=5,
        default=None,
        help="Optional 5-class weights [w0, w1, w2, w3, w4].",
    )
    parser.add_argument(
        "--min-best-epoch",
        type=int,
        default=None,
        help="Earliest epoch eligible to become best checkpoint.",
    )
    parser.add_argument(
        "--selection-metric",
        default=None,
        choices=["macro_f1", "oa_balanced", "oa_balanced_clean", "val_loss", "val_balanced_loss"],
        help="Validation objective for checkpoint selection.",
    )
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay.")
    parser.add_argument(
        "--optimizer",
        default=None,
        choices=["adam", "adamw"],
        help="Optimizer choice ('adam' or 'adamw').",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        choices=["vgg19_bn", "vgg19", "vgg19_sp", "resnet50", "densenet121", "convnext_tiny", "efficientnet_v2_s"],
        help="Override model backbone without mutating shared classifier YAML.",
    )
    parser.add_argument("--scale-jitter", type=float, default=0.07, help="Fast GPU loader scale jitter.")
    parser.add_argument("--translate-x", type=float, default=0.05, help="Fast GPU loader horizontal shift fraction.")
    parser.add_argument("--translate-y", type=float, default=None, help="Fast GPU loader vertical shift fraction.")
    parser.add_argument("--cutout-prob", type=float, default=0.0, help="Fast GPU loader random erasing cutout prob.")
    parser.add_argument("--brightness-jitter", type=float, default=0.10, help="Fast GPU loader brightness jitter.")
    parser.add_argument("--contrast-jitter", type=float, default=0.12, help="Fast GPU loader contrast jitter.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Optional MixUp Beta(alpha, alpha) regularization.")
    parser.add_argument("--ordinal-weight", type=float, default=0.0, help="Optional expected-KL ordinal progression loss weight.")
    parser.add_argument("--stochastic-depth", type=float, default=0.1, help="Drop-path probability for ConvNeXt backbones.")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="Optional EMA decay rate (e.g. 0.996).")
    parser.add_argument("--tta-views", type=int, default=1, choices=[1, 2, 5], help="Spatial/scale TTA views at eval.")
    parser.add_argument(
        "--hierarchical-oa",
        action="store_true",
        help="Enforce binary-OA probability coherence when KL3/KL4 split mass outweighs KL<=2.",
    )
    parser.add_argument(
        "--aux-subscore-weight",
        type=float,
        default=0.0,
        help="Optional train-only auxiliary OARSI OP/JSN/ER supervision weight.",
    )
    parser.add_argument("--oa-bce-weight", type=float, default=0.0, help="Stage-1 binary OA margin BCE loss weight.")
    parser.add_argument("--oa-pos-weight", type=float, default=1.6, help="Positive weight for Stage-1 binary OA BCE.")
    parser.add_argument("--severe-pair-weight", type=float, default=0.0, help="Stage-2 KL4-vs-KL3 pairwise BCE weight.")
    parser.add_argument("--severe-pos-weight", type=float, default=0.85, help="Positive weight for KL4 inside KL3/KL4 pair loss.")
    parser.add_argument(
        "--scheduler",
        default=None,
        choices=["none", "plateau", "cosine"],
        help="Optional LR scheduler ('none', 'plateau', or 'cosine').",
    )
    parser.add_argument("--lr-factor", type=float, default=None, help="ReduceLROnPlateau factor.")
    parser.add_argument("--lr-patience", type=int, default=None, help="ReduceLROnPlateau patience.")
    parser.add_argument("--min-lr", type=float, default=None, help="Minimum LR for plateau scheduler.")
    parser.add_argument(
        "--fast-gpu-loader",
        action="store_true",
        help="Preload images onto target device and run flip/affine batch augmentations on GPU.",
    )
    parser.add_argument(
        "--seed-tagged",
        action="store_true",
        help="Always include _seed{seed} in output JSON filename.",
    )
    parser.add_argument(
        "--summarize-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Aggregate existing {tag}_seed{s}_test_metrics.json files into multiseed summary without training.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=3600,
        help="Max seconds --summarize-seeds waits for sibling GPU workers to finish.",
    )
    parser.add_argument("--tag", default="baseline", help="Run identifier for saving metrics.")
    return parser.parse_args()


def load_synthetic_items(
    manifest_csv: str | Path,
    max_synthetic: int | None,
    seed: int,
) -> list[tuple[Path, int]]:
    """Load synthetic augmentation samples and verify zero test-split leakage."""
    df = pd.read_csv(manifest_csv)
    if "source_split" in df.columns and (df["source_split"] == "test").any():
        raise AssertionError("Refusing synthetic manifest derived from 'test' split patients!")
    if max_synthetic is not None and max_synthetic < len(df):
        df = df.sample(n=max_synthetic, random_state=seed).reset_index(drop=True)
    return [(Path(row["image_path"]), int(row["target_kl"])) for row in df.to_dict("records")]


def build_classification_loaders(
    manifest: pd.DataFrame,
    images_dir: str | Path,
    joints: list[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    synthetic_items: list[tuple[Path, int]] | None = None,
    oversample: dict[int, int] | None = None,
    synthetic_oversample: dict[int, int] | None = None,
    include_test: bool = True,
    pre_resize: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Construct train, val, and optional test DataLoaders with optional RAM pre-resize."""
    pre_sz = int(image_size) if pre_resize else None
    pin_mem = torch.cuda.is_available()
    persistent = num_workers > 0

    train_ds = JointClassificationDataset(
        manifest_df=manifest,
        images_dir=images_dir,
        split="train",
        joints=joints,
        transform=build_classifier_transform(
            image_size=image_size, is_train=True, skip_resize=pre_resize
        ),
        synthetic_items=synthetic_items,
        oversample_multipliers=oversample,
        synthetic_oversample_multipliers=synthetic_oversample,
        pre_resize_size=pre_sz,
    )
    val_ds = JointClassificationDataset(
        manifest_df=manifest,
        images_dir=images_dir,
        split="val",
        joints=joints,
        transform=build_classifier_transform(
            image_size=image_size, is_train=False, skip_resize=pre_resize
        ),
        pre_resize_size=pre_sz,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
    )

    test_loader = None
    if include_test:
        test_ds = JointClassificationDataset(
            manifest_df=manifest,
            images_dir=images_dir,
            split="test",
            joints=joints,
            transform=build_classifier_transform(
                image_size=image_size, is_train=False, skip_resize=pre_resize
            ),
            pre_resize_size=pre_sz,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent,
        )
    return train_loader, val_loader, test_loader


def _resolve_focal_settings(
    args: argparse.Namespace,
    train_cfg: dict[str, Any],
) -> tuple[float, list[float] | None, int, str]:
    """Merge focal hyperparameters from YAML, optional Hyperopt JSON, and CLI flags."""
    focal_gamma = float(train_cfg["focal_gamma"])
    class_weights: list[float] | None = (
        [float(w) for w in train_cfg["class_weights"]]
        if train_cfg.get("class_weights") is not None
        else None
    )
    min_best_epoch = int(train_cfg.get("min_best_epoch", 1))
    selection_metric = str(train_cfg.get("selection_metric", "macro_f1"))

    if args.focal_config_json is not None:
        cfg_path = resolve_project_path(args.focal_config_json, _ROOT)
        tuned = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "focal_gamma" in tuned:
            focal_gamma = float(tuned["focal_gamma"])
        if tuned.get("class_weights") is not None:
            class_weights = [float(w) for w in tuned["class_weights"]]
        if "min_best_epoch" in tuned:
            min_best_epoch = int(tuned["min_best_epoch"])
        if "selection_metric" in tuned:
            selection_metric = str(tuned["selection_metric"])

    if args.focal_gamma is not None:
        focal_gamma = float(args.focal_gamma)
    if args.class_weights is not None:
        class_weights = [float(w) for w in args.class_weights]
    if args.min_best_epoch is not None:
        min_best_epoch = int(args.min_best_epoch)
    if args.selection_metric is not None:
        selection_metric = str(args.selection_metric)
    return focal_gamma, class_weights, min_best_epoch, selection_metric


def _summarize_multiseed(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean and sample standard deviation across multi-seed evaluation reports."""
    ddof = 1 if len(reports) > 1 else 0
    metric_keys = [
        "accuracy",
        "macro_f1",
        "binary_oa_sensitivity",
        "binary_oa_precision",
        "binary_oa_f1",
        "binary_oa_roc_auc",
    ]
    summary: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        vals = [float(r["test_metrics"][key]) for r in reports if r["test_metrics"].get(key) is not None]
        if vals:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=ddof)),
            }

    per_class: dict[str, dict[str, float]] = {}
    for grade in reports[0]["test_metrics"]["per_class_f1"]:
        vals = [float(r["test_metrics"]["per_class_f1"][grade]) for r in reports]
        per_class[grade] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=ddof)),
        }
    return {
        "tag": reports[0]["tag"],
        "seeds": [int(r["seed"]) for r in reports],
        "num_runs": len(reports),
        "metrics": summary,
        "per_class_f1": per_class,
    }


def _wait_and_summarize_seeds(
    out_dir: Path,
    tag: str,
    seeds: list[int],
    timeout_sec: int,
) -> dict[str, Any]:
    """Wait until per-seed reports exist and write aggregated multi-seed summary JSON."""
    expected = [(int(s), out_dir / f"{tag}_seed{int(s)}_test_metrics.json") for s in seeds]
    deadline = time.monotonic() + max(0, int(timeout_sec))
    while True:
        missing = [p for _s, p in expected if not p.is_file()]
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for multi-seed reports: {[str(m) for m in missing]}"
            )
        time.sleep(5)

    reports = [json.loads(p.read_text(encoding="utf-8")) for _s, p in expected]
    summary = _summarize_multiseed(reports)
    save_json(summary, out_dir / f"{tag}_multiseed_summary.json")
    print("\n=== MULTI-SEED SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml_config(resolve_project_path(args.data_config, _ROOT))
    cls_cfg = load_yaml_config(resolve_project_path(args.config, _ROOT))

    out_dir_raw = args.output_dir if args.output_dir is not None else cls_cfg["experiment"]["output_dir"]
    out_dir = ensure_dir(resolve_project_path(out_dir_raw, _ROOT))
    if args.summarize_seeds:
        _wait_and_summarize_seeds(out_dir, args.tag, list(args.summarize_seeds), args.wait_timeout)
        return

    default_seed = int(args.seed if args.seed is not None else cls_cfg["experiment"].get("seed", 42))
    run_seeds = [int(s) for s in args.seeds] if args.seeds else [default_seed]

    manifest, images_dir = load_manifest_from_config(data_cfg, _ROOT)
    cls_data = cls_cfg["data"]
    joints = cls_data.get("joints", data_cfg["dataset"]["joints"])
    image_size = int(cls_data["image_size"])

    synthetic_items = None
    if args.synthetic_manifest is not None:
        syn_path = resolve_project_path(args.synthetic_manifest, _ROOT)
        synthetic_items = load_synthetic_items(syn_path, args.max_synthetic, run_seeds[0])

    oversample_cfg = {
        int(k): int(v) for k, v in cls_data.get("oversample_multipliers", {}).items()
    }
    if args.oversample_kl3 is not None:
        oversample_cfg[3] = int(args.oversample_kl3)
    if args.oversample_kl4 is not None:
        oversample_cfg[4] = int(args.oversample_kl4)
    oversample = {k: v for k, v in oversample_cfg.items() if v > 1} or None

    syn_repeat_cfg: dict[int, int] = {}
    if args.synthetic_repeat_kl3 is not None:
        syn_repeat_cfg[3] = int(args.synthetic_repeat_kl3)
    if args.synthetic_repeat_kl4 is not None:
        syn_repeat_cfg[4] = int(args.synthetic_repeat_kl4)
    synthetic_oversample = syn_repeat_cfg or None

    aux_subscore_weight = float(args.aux_subscore_weight)
    batch_size = int(args.batch_size if args.batch_size is not None else cls_data["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.fast_gpu_loader:
        train_loader, val_loader, test_loader = build_fast_classification_loaders(
            manifest=manifest,
            images_dir=images_dir,
            joints=joints,
            image_size=image_size,
            batch_size=batch_size,
            device=device,
            synthetic_items=synthetic_items,
            oversample=oversample,
            synthetic_oversample=synthetic_oversample,
            include_test=True,
            max_translate_frac=args.translate_x,
            max_translate_y_frac=args.translate_y,
            max_scale_jitter=args.scale_jitter,
            brightness_jitter=args.brightness_jitter,
            contrast_jitter=args.contrast_jitter,
            cutout_prob=args.cutout_prob,
            return_aux=(aux_subscore_weight > 0.0),
        )
    else:
        train_loader, val_loader, test_loader = build_classification_loaders(
            manifest=manifest,
            images_dir=images_dir,
            joints=joints,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=int(cls_data["num_workers"]),
            synthetic_items=synthetic_items,
            oversample=oversample,
            synthetic_oversample=synthetic_oversample,
            include_test=True,
            pre_resize=True,
        )
    assert test_loader is not None

    model_cfg = cls_cfg["model"]
    train_cfg = cls_cfg["training"]
    backbone = str(args.backbone if args.backbone is not None else model_cfg["backbone"])
    optimizer_type = str(args.optimizer if args.optimizer is not None else train_cfg.get("optimizer", "adamw"))
    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    lr = float(args.lr if args.lr is not None else train_cfg["lr"])
    weight_decay = float(
        args.weight_decay if args.weight_decay is not None else train_cfg["weight_decay"]
    )
    loss_type = str(args.loss_type if args.loss_type is not None else train_cfg["loss_type"])
    patience = int(args.patience if args.patience is not None else train_cfg.get("patience", 12))
    scheduler_type = str(
        args.scheduler if args.scheduler is not None else train_cfg.get("scheduler", "none")
    )
    lr_factor = float(
        args.lr_factor if args.lr_factor is not None else train_cfg.get("lr_factor", 0.5)
    )
    lr_patience = int(
        args.lr_patience if args.lr_patience is not None else train_cfg.get("lr_patience", 5)
    )
    min_lr = float(args.min_lr if args.min_lr is not None else train_cfg.get("min_lr", 1e-6))
    focal_gamma, class_weights, min_best_epoch, selection_metric = _resolve_focal_settings(
        args, train_cfg
    )

    reports: list[dict[str, Any]] = []
    for run_seed in run_seeds:
        seed_everything(run_seed)
        model = build_classifier(
            backbone=backbone,
            num_classes=int(model_cfg["num_classes"]),
            pretrained=bool(model_cfg["pretrained"]),
            dropout=float(model_cfg["dropout"]),
            stochastic_depth_prob=float(args.stochastic_depth),
        )
        trained_model, history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            num_classes=int(model_cfg["num_classes"]),
            oa_threshold_kl=int(train_cfg["oa_threshold_kl"]),
            patience=patience,
            min_best_epoch=min_best_epoch,
            selection_metric=selection_metric,
            scheduler_type=scheduler_type,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            min_lr=min_lr,
            optimizer_type=optimizer_type,
            mixup_alpha=float(args.mixup_alpha),
            ordinal_weight=float(args.ordinal_weight),
            ema_decay=float(args.ema_decay),
            aux_subscore_weight=aux_subscore_weight,
            oa_bce_weight=float(args.oa_bce_weight),
            oa_pos_weight=float(args.oa_pos_weight),
            severe_pair_weight=float(args.severe_pair_weight),
            severe_pos_weight=float(args.severe_pos_weight),
            tta_views=int(args.tta_views),
            hierarchical_oa=bool(args.hierarchical_oa),
        )
        if args.save_checkpoint is not None and run_seed == run_seeds[0]:
            ckpt_out = resolve_project_path(args.save_checkpoint, _ROOT)
            ensure_dir(ckpt_out.parent)
            torch.save(trained_model.state_dict(), ckpt_out)

        if args.skip_test_eval:
            print(f"Saved checkpoint for seed={run_seed}; skipped test evaluation (--skip-test-eval).")
            continue

        test_metrics = evaluate_classifier(
            model=trained_model,
            loader=test_loader,
            device=device,
            num_classes=int(model_cfg["num_classes"]),
            oa_threshold_kl=int(train_cfg["oa_threshold_kl"]),
            tta_views=int(args.tta_views),
            hierarchical_oa=bool(args.hierarchical_oa),
        )

        eff_min = max(1, min(int(min_best_epoch), len(history)))
        eligible = [row for row in history if int(row["epoch"]) >= eff_min]
        best_hist = max(eligible, key=lambda r: (float(r["val"]["selection_score"]), -int(r["epoch"])))

        report = {
            "tag": args.tag,
            "backbone": backbone,
            "optimizer": optimizer_type,
            "seed": run_seed,
            "lr": lr,
            "weight_decay": weight_decay,
            "mixup_alpha": float(args.mixup_alpha),
            "ordinal_weight": float(args.ordinal_weight),
            "stochastic_depth": float(args.stochastic_depth),
            "ema_decay": float(args.ema_decay),
            "tta_views": int(args.tta_views),
            "hierarchical_oa": bool(args.hierarchical_oa),
            "aux_subscore_weight": aux_subscore_weight,
            "oa_bce_weight": float(args.oa_bce_weight),
            "severe_pair_weight": float(args.severe_pair_weight),
            "scheduler": scheduler_type,
            "oversample": oversample_cfg,
            "synthetic_repeat": syn_repeat_cfg,
            "focal_gamma": focal_gamma,
            "class_weights": class_weights,
            "min_best_epoch": min_best_epoch,
            "selection_metric": selection_metric,
            "best_epoch": int(best_hist["epoch"]),
            "best_val_metrics": best_hist["val"],
            "synthetic_count": len(synthetic_items) if synthetic_items else 0,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "test_metrics": test_metrics,
        }
        reports.append(report)
        if len(run_seeds) == 1 and not args.seed_tagged:
            save_json(report, out_dir / f"{args.tag}_test_metrics.json")
        else:
            save_json(report, out_dir / f"{args.tag}_seed{run_seed}_test_metrics.json")
            if len(run_seeds) > 1 and run_seed == run_seeds[0]:
                save_json(report, out_dir / f"{args.tag}_test_metrics.json")
        print(json.dumps(report, indent=2))

    if len(reports) > 1:
        summary = _summarize_multiseed(reports)
        save_json(summary, out_dir / f"{args.tag}_multiseed_summary.json")
        print("\n=== MULTI-SEED SUMMARY ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
