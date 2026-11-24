"""Asynchronous multi-GPU Hyperopt TPE search for Focal Loss on val split only."""

import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from hyperopt import JOB_STATE_DONE, JOB_STATE_RUNNING, STATUS_OK, Trials, fmin, hp, tpe

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.run_downstream_eval import build_classification_loaders, load_synthetic_items  # noqa: E402
from hand_oa.data import load_manifest_from_config  # noqa: E402
from hand_oa.evaluation import (  # noqa: E402
    build_classifier,
    build_fast_classification_loaders,
    train_classifier,
)
from hand_oa.utils import (  # noqa: E402
    ensure_dir,
    load_yaml_config,
    resolve_project_path,
    save_json,
    seed_everything,
)

_SPACE_KEYS = ("focal_gamma", "base_alpha", "kl3_alpha_mult", "kl4_alpha_mult")
_SPACE_BOUNDS = {
    "focal_gamma": (1.0, 5.0),
    "base_alpha": (0.5, 3.0),
    "kl3_alpha_mult": (1.0, 3.0),
    "kl4_alpha_mult": (1.0, 3.0),
}


class _ParamsSampled(Exception):
    """Internal control-flow signal returning one TPE parameter sample cleanly."""

    def __init__(self, params: dict[str, float]) -> None:
        super().__init__("sampled")
        self.params = {k: float(params[k]) for k in _SPACE_KEYS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search focal gamma and class weights via Hyperopt TPE on val split only."
    )
    parser.add_argument("--data-config", default="configs/data_filtered.yaml")
    parser.add_argument("--config", default="configs/classifier_filtered.yaml")
    parser.add_argument(
        "--synthetic-manifest",
        default="outputs/cyclegan_filtered/synthetic_kl3/synthetic_manifest_filtered.csv",
    )
    parser.add_argument("--max-synthetic", type=int, default=None)
    parser.add_argument("--oversample-kl3", type=int, default=5)
    parser.add_argument("--oversample-kl4", type=int, default=10)
    parser.add_argument("--max-trials", type=int, default=30)
    parser.add_argument("--trial-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-best-epoch", type=int, default=8)
    parser.add_argument(
        "--selection-metric",
        default="oa_balanced",
        choices=["macro_f1", "oa_balanced"],
    )
    parser.add_argument("--output-dir", default="outputs/evaluation_v5_hyperopt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None, help="Override classifier batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Optional worker identifier for multi-GPU array logs (defaults to SGE_TASK_ID).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume completed trials from existing hyperopt_trials.csv without duplicates.",
    )
    parser.add_argument(
        "--fast-gpu-loader",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload images on target compute device with vectorized GPU augmentation.",
    )
    return parser.parse_args()


@contextmanager
def _file_lock(lock_dir: Path, timeout_sec: float = 120.0) -> Iterator[None]:
    """Cross-node atomic mkdir lock safe over shared GPFS/NFS cluster storage."""
    deadline = time.monotonic() + timeout_sec
    acquired = False
    while not acquired:
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            acquired = True
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for HPO coordination lock: {lock_dir}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if lock_dir.is_dir():
            lock_dir.rmdir()


def extract_best_epoch_metrics(
    history: list[dict[str, Any]],
    min_best_epoch: int,
) -> tuple[int, dict[str, Any]]:
    """Select the winning epoch entry matching train_classifier checkpoint rules."""
    effective_min = max(1, min(int(min_best_epoch), len(history)))
    eligible = [row for row in history if int(row["epoch"]) >= effective_min]
    best_row = max(eligible, key=lambda r: (float(r["val"]["selection_score"]), -int(r["epoch"])))
    return int(best_row["epoch"]), dict(best_row["val"])


def _row_to_val_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct validation metrics dictionary from a saved CSV trial row."""
    return {
        "selection_score": float(row["val_selection_score"]),
        "macro_f1": float(row["val_macro_f1"]),
        "accuracy": float(row["val_accuracy"]),
        "per_class_f1": {
            "KL1": float(row["val_kl1_f1"]),
            "KL2": float(row["val_kl2_f1"]),
            "KL3": float(row["val_kl3_f1"]),
            "KL4": float(row["val_kl4_f1"]),
        },
        "binary_oa_f1": float(row["val_binary_oa_f1"]),
        "binary_oa_sensitivity": float(row["val_binary_oa_sensitivity"]),
        "binary_oa_precision": float(row["val_binary_oa_precision"]),
    }


def _read_completed_records(trials_csv: Path) -> list[dict[str, Any]]:
    if not trials_csv.is_file():
        return []
    df = pd.read_csv(trials_csv)
    return [] if df.empty else df.to_dict("records")


def _write_csv_atomic(records: list[dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    pd.DataFrame(records).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _read_inflight(inflight_path: Path, current_job_id: str) -> list[dict[str, Any]]:
    if not inflight_path.is_file():
        return []
    items = json.loads(inflight_path.read_text(encoding="utf-8"))
    if current_job_id != "local":
        # Prune stale claims left behind by terminated earlier qsub jobs
        items = [it for it in items if str(it.get("job_id", "")) == current_job_id]
    return items


def _write_inflight_atomic(items: list[dict[str, Any]], inflight_path: Path) -> None:
    tmp = inflight_path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
    os.replace(tmp, inflight_path)


def _build_hyperopt_trials(
    completed: list[dict[str, Any]],
    inflight: list[dict[str, Any]] | None = None,
) -> Trials:
    """Build Hyperopt Trials containing both completed and active inflight trials."""
    trials = Trials()
    tid = 0
    for row in completed:
        trials.insert_trial_doc(
            {
                "state": JOB_STATE_DONE,
                "tid": tid,
                "spec": None,
                "result": {
                    "loss": -float(row["val_selection_score"]),
                    "status": STATUS_OK,
                    "trial_record": row,
                    "best_val": _row_to_val_summary(row),
                },
                "misc": {
                    "tid": tid,
                    "cmd": ("domain_attachment", "FMinIter_Domain"),
                    "idxs": {k: [tid] for k in _SPACE_KEYS},
                    "vals": {k: [float(row[k])] for k in _SPACE_KEYS},
                },
                "exp_key": None,
                "owner": None,
                "version": 0,
                "book_time": None,
                "refresh_time": None,
            }
        )
        tid += 1

    for entry in inflight or []:
        params = entry["params"]
        trials.insert_trial_doc(
            {
                "state": JOB_STATE_RUNNING,
                "tid": tid,
                "spec": None,
                "result": {"status": "new"},
                "misc": {
                    "tid": tid,
                    "cmd": ("domain_attachment", "FMinIter_Domain"),
                    "idxs": {k: [tid] for k in _SPACE_KEYS},
                    "vals": {k: [float(params[k])] for k in _SPACE_KEYS},
                },
                "exp_key": None,
                "owner": None,
                "version": 0,
                "book_time": None,
                "refresh_time": None,
            }
        )
        tid += 1

    trials.refresh()
    return trials


def load_resumed_trials(trials_csv: Path) -> tuple[Trials, list[dict[str, Any]]]:
    """Rebuild Hyperopt Trials history from completed rows in hyperopt_trials.csv."""
    records = _read_completed_records(trials_csv)
    return _build_hyperopt_trials(records), records


def _is_near_duplicate(candidate: dict[str, float], existing_params: list[dict[str, float]]) -> bool:
    for prev in existing_params:
        max_norm_diff = max(
            abs(float(candidate[k]) - float(prev[k])) / (_SPACE_BOUNDS[k][1] - _SPACE_BOUNDS[k][0])
            for k in _SPACE_KEYS
        )
        if max_norm_diff < 1e-3:
            return True
    return False


def _sample_next_params(
    completed: list[dict[str, Any]],
    inflight: list[dict[str, Any]],
    space: dict[str, Any],
    base_seed: int,
) -> dict[str, float]:
    """Draw one non-duplicate TPE candidate accounting for completed and inflight trials."""
    existing_params = [{k: float(r[k]) for k in _SPACE_KEYS} for r in completed] + [
        {k: float(it["params"][k]) for k in _SPACE_KEYS} for it in inflight
    ]
    n_accounted = len(existing_params)

    for attempt in range(8):
        trials = _build_hyperopt_trials(completed, inflight)
        rng = np.random.default_rng(int(base_seed) + n_accounted + attempt * 97)
        try:
            fmin(
                fn=lambda p: (_ for _ in ()).throw(_ParamsSampled(p)),
                space=space,
                algo=tpe.suggest,
                max_evals=len(trials.trials) + 1,
                trials=trials,
                rstate=rng,
                catch_eval_exceptions=False,
                show_progressbar=False,
            )
        except _ParamsSampled as sig:
            if not _is_near_duplicate(sig.params, existing_params) or attempt == 7:
                return sig.params
    raise RuntimeError("Failed to sample next Hyperopt parameter candidate.")


def _write_best_config(
    completed: list[dict[str, Any]],
    out_dir: Path,
    max_trials: int,
    trial_epochs: int,
    patience: int,
    min_best_epoch: int,
    selection_metric: str,
    seed: int,
) -> dict[str, Any]:
    best_trial = max(completed, key=lambda r: float(r["val_selection_score"]))
    best_config = {
        "method": "hyperopt_tpe",
        "max_trials": int(max_trials),
        "completed_trials": len(completed),
        "trial_epochs": int(trial_epochs),
        "patience": int(patience),
        "min_best_epoch": int(min_best_epoch),
        "selection_metric": selection_metric,
        "seed": int(seed),
        "focal_gamma": float(best_trial["focal_gamma"]),
        "base_alpha": float(best_trial["base_alpha"]),
        "kl3_alpha_mult": float(best_trial["kl3_alpha_mult"]),
        "kl4_alpha_mult": float(best_trial["kl4_alpha_mult"]),
        "class_weights": [
            float(best_trial["weight_kl0_to_2"]),
            float(best_trial["weight_kl0_to_2"]),
            float(best_trial["weight_kl0_to_2"]),
            float(best_trial["weight_kl3"]),
            float(best_trial["weight_kl4"]),
        ],
        "best_trial_id": int(best_trial["trial_id"]),
        "best_epoch": int(best_trial["best_epoch"]),
        "best_val_selection_score": float(best_trial["val_selection_score"]),
        "best_val_metrics": _row_to_val_summary(best_trial),
    }
    tmp = (out_dir / "best_focal_config.json").with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    os.replace(tmp, out_dir / "best_focal_config.json")
    return best_config


def run_hyperopt_search(
    train_loader: Any,
    val_loader: Any,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    max_trials: int = 30,
    trial_epochs: int = 30,
    patience: int = 8,
    min_best_epoch: int = 8,
    selection_metric: str = "oa_balanced",
    seed: int = 42,
    resume: bool = True,
    worker_id: str | None = None,
) -> dict[str, Any]:
    """Run lock-coordinated (1 or more GPUs) TPE Bayesian search over focal loss params."""
    ensure_dir(out_dir)
    space = {k: hp.uniform(k, lo, hi) for k, (lo, hi) in _SPACE_BOUNDS.items()}
    trials_csv = out_dir / "hyperopt_trials.csv"
    inflight_path = out_dir / "hyperopt_inflight.json"
    lock_dir = out_dir / ".hyperopt_lock"
    current_job_id = str(os.environ.get("JOB_ID", "local"))
    worker_tag = str(worker_id or os.environ.get("SGE_TASK_ID", "1"))

    if not resume:
        with _file_lock(lock_dir):
            if trials_csv.is_file():
                trials_csv.unlink()
            if inflight_path.is_file():
                inflight_path.unlink()

    with _file_lock(lock_dir):
        existing = _read_completed_records(trials_csv)
        if existing:
            print(
                f"[Hyperopt Worker w{worker_tag}] Found {len(existing)} completed trials in {trials_csv}; "
                f"target={max_trials}."
            )

    local_seq = 0
    while True:
        with _file_lock(lock_dir):
            completed = _read_completed_records(trials_csv)
            inflight = _read_inflight(inflight_path, current_job_id)

            if len(completed) >= max_trials:
                if inflight:
                    _write_inflight_atomic([], inflight_path)
                return _write_best_config(
                    completed,
                    out_dir,
                    max_trials,
                    trial_epochs,
                    patience,
                    min_best_epoch,
                    selection_metric,
                    seed,
                )

            if len(completed) + len(inflight) >= max_trials:
                action = "wait"
                claim_id = ""
                params = {}
            else:
                local_seq += 1
                params = _sample_next_params(completed, inflight, space, seed)
                claim_id = f"{current_job_id}_w{worker_tag}_{os.getpid()}_{local_seq}"
                inflight.append({"claim_id": claim_id, "job_id": current_job_id, "params": params})
                _write_inflight_atomic(inflight, inflight_path)
                action = "run"
                est_num = len(completed) + len(inflight)

        if action == "wait":
            time.sleep(5)
            continue

        focal_gamma = float(params["focal_gamma"])
        base_alpha = float(params["base_alpha"])
        kl3_mult = float(params["kl3_alpha_mult"])
        kl4_mult = float(params["kl4_alpha_mult"])
        class_weights = [
            base_alpha,
            base_alpha,
            base_alpha,
            base_alpha * kl3_mult,
            base_alpha * kl4_mult,
        ]
        print(
            f"\n[Hyperopt Worker w{worker_tag} | Claim #{est_num}/{max_trials}] "
            f"gamma={focal_gamma:.4f}, base_alpha={base_alpha:.4f}, "
            f"kl3_mult={kl3_mult:.4f}, kl4_mult={kl4_mult:.4f}"
        )

        try:
            seed_everything(seed)
            model = build_classifier(
                backbone=str(model_cfg["backbone"]),
                num_classes=int(model_cfg["num_classes"]),
                pretrained=bool(model_cfg["pretrained"]),
                dropout=float(model_cfg["dropout"]),
            )
            _, history = train_classifier(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=trial_epochs,
                lr=float(train_cfg["lr"]),
                weight_decay=float(train_cfg["weight_decay"]),
                loss_type="focal",
                focal_gamma=focal_gamma,
                class_weights=class_weights,
                num_classes=int(model_cfg["num_classes"]),
                oa_threshold_kl=int(train_cfg["oa_threshold_kl"]),
                patience=patience,
                min_best_epoch=min_best_epoch,
                selection_metric=selection_metric,
            )
            best_ep, best_val = extract_best_epoch_metrics(history, min_best_epoch=min_best_epoch)
        except Exception:
            with _file_lock(lock_dir):
                inflight_now = _read_inflight(inflight_path, current_job_id)
                _write_inflight_atomic(
                    [it for it in inflight_now if it.get("claim_id") != claim_id],
                    inflight_path,
                )
            raise

        with _file_lock(lock_dir):
            inflight_now = _read_inflight(inflight_path, current_job_id)
            _write_inflight_atomic(
                [it for it in inflight_now if it.get("claim_id") != claim_id],
                inflight_path,
            )
            completed = _read_completed_records(trials_csv)
            trial_id = len(completed) + 1
            row = {
                "trial_id": trial_id,
                "focal_gamma": round(focal_gamma, 6),
                "base_alpha": round(base_alpha, 6),
                "kl3_alpha_mult": round(kl3_mult, 6),
                "kl4_alpha_mult": round(kl4_mult, 6),
                "weight_kl0_to_2": round(base_alpha, 6),
                "weight_kl3": round(class_weights[3], 6),
                "weight_kl4": round(class_weights[4], 6),
                "best_epoch": best_ep,
                "val_selection_score": float(best_val["selection_score"]),
                "val_macro_f1": float(best_val["macro_f1"]),
                "val_accuracy": float(best_val["accuracy"]),
                "val_kl1_f1": float(best_val["per_class_f1"]["KL1"]),
                "val_kl2_f1": float(best_val["per_class_f1"]["KL2"]),
                "val_kl3_f1": float(best_val["per_class_f1"]["KL3"]),
                "val_kl4_f1": float(best_val["per_class_f1"]["KL4"]),
                "val_binary_oa_f1": float(best_val["binary_oa_f1"]),
                "val_binary_oa_sensitivity": float(best_val["binary_oa_sensitivity"]),
                "val_binary_oa_precision": float(best_val["binary_oa_precision"]),
            }
            completed.append(row)
            _write_csv_atomic(completed, trials_csv)
            best_cfg = _write_best_config(
                completed,
                out_dir,
                max_trials,
                trial_epochs,
                patience,
                min_best_epoch,
                selection_metric,
                seed,
            )
            print(
                f"[Hyperopt Worker w{worker_tag}] Recorded Trial {trial_id}/{max_trials} "
                f"(score={row['val_selection_score']:.4f}, best_overall={best_cfg['best_val_selection_score']:.4f})"
            )


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml_config(resolve_project_path(args.data_config, _ROOT))
    cls_cfg = load_yaml_config(resolve_project_path(args.config, _ROOT))

    manifest, images_dir = load_manifest_from_config(data_cfg, _ROOT, overwrite=False)
    cls_data = cls_cfg["data"]
    joints = cls_data.get("joints", data_cfg["dataset"]["joints"])
    image_size = int(cls_data["image_size"])

    synthetic_items = None
    if args.synthetic_manifest:
        syn_path = resolve_project_path(args.synthetic_manifest, _ROOT)
        synthetic_items = load_synthetic_items(syn_path, args.max_synthetic, args.seed)

    oversample = {
        k: v
        for k, v in {3: int(args.oversample_kl3), 4: int(args.oversample_kl4)}.items()
        if v > 1
    } or None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(args.batch_size if args.batch_size is not None else cls_data["batch_size"])
    train_cfg = dict(cls_cfg["training"])
    if args.lr is not None:
        train_cfg["lr"] = float(args.lr)

    # Test split is explicitly excluded during Hyperopt tuning to prevent any test leakage
    if args.fast_gpu_loader:
        train_loader, val_loader, _ = build_fast_classification_loaders(
            manifest=manifest,
            images_dir=images_dir,
            joints=joints,
            image_size=image_size,
            batch_size=batch_size,
            device=device,
            synthetic_items=synthetic_items,
            oversample=oversample,
            include_test=False,
        )
    else:
        train_loader, val_loader, _ = build_classification_loaders(
            manifest=manifest,
            images_dir=images_dir,
            joints=joints,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=int(cls_data["num_workers"]),
            synthetic_items=synthetic_items,
            oversample=oversample,
            include_test=False,
            pre_resize=True,
        )

    out_dir = ensure_dir(resolve_project_path(args.output_dir, _ROOT))
    best_config = run_hyperopt_search(
        train_loader=train_loader,
        val_loader=val_loader,
        model_cfg=cls_cfg["model"],
        train_cfg=train_cfg,
        device=device,
        out_dir=out_dir,
        max_trials=args.max_trials,
        trial_epochs=args.trial_epochs,
        patience=args.patience,
        min_best_epoch=args.min_best_epoch,
        selection_metric=args.selection_metric,
        seed=args.seed,
        resume=args.resume,
        worker_id=args.worker_id,
    )
    print("\n=== BEST DISCOVERED HYPEROPT FOCAL CONFIG ===")
    print(json.dumps(best_config, indent=2))


if __name__ == "__main__":
    main()
