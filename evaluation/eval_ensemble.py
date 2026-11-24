"""Evaluate validation-selected multi-backbone Model 5 ensembles cleanly without test leakage."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hand_oa.data import load_manifest_from_config
from hand_oa.evaluation import build_classifier, build_fast_classification_loaders, compute_selection_score
from hand_oa.evaluation.classifier import decode_predictions_from_probs
from hand_oa.evaluation.metrics import compute_classification_metrics
from hand_oa.utils import ensure_dir, load_yaml_config, resolve_project_path, save_json


@torch.no_grad()
def _predict_probs(backbone: str, ckpt_path: Path, loader, device: torch.device) -> np.ndarray:
    model = build_classifier(backbone=backbone, num_classes=5, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    probs_list = []
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        probs_list.append(torch.softmax(model(images), dim=1).cpu().numpy())
    return np.concatenate(probs_list, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-selected multi-backbone ensemble evaluator.")
    parser.add_argument("--data-config", default="configs/data_filtered.yaml")
    parser.add_argument("--config", default="configs/classifier_filtered.yaml")
    parser.add_argument("--hierarchical-oa", action="store_true")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Pairs in backbone:checkpoint_path format.",
    )
    args = parser.parse_args()

    data_cfg = load_yaml_config(resolve_project_path(args.data_config, _ROOT))
    cls_cfg = load_yaml_config(resolve_project_path(args.config, _ROOT))
    manifest, images_dir = load_manifest_from_config(data_cfg, _ROOT)
    joints = cls_cfg["data"].get("joints", data_cfg["dataset"]["joints"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, test_loader = build_fast_classification_loaders(
        manifest=manifest,
        images_dir=images_dir,
        joints=joints,
        image_size=int(cls_cfg["data"]["image_size"]),
        batch_size=64,
        device=device,
        include_test=True,
    )
    assert test_loader is not None

    y_val = val_loader.labels.cpu().numpy()
    y_test = test_loader.labels.cpu().numpy()

    specs = []
    for item in args.models:
        bb, ckpt = item.split(":", 1)
        specs.append((bb, Path(ckpt)))

    val_probs = [_predict_probs(bb, ckpt, val_loader, device) for bb, ckpt in specs]
    test_probs = [_predict_probs(bb, ckpt, test_loader, device) for bb, ckpt in specs]

    for (bb, ckpt), vp, tp in zip(specs, val_probs, test_probs):
        for h in (False, True) if args.hierarchical_oa else (False,):
            v_pred = decode_predictions_from_probs(torch.from_numpy(vp), h).numpy()
            t_pred = decode_predictions_from_probs(torch.from_numpy(tp), h).numpy()
            vm = compute_classification_metrics(y_val, v_pred, vp)
            tm = compute_classification_metrics(y_test, t_pred, tp)
            print(
                f"Single {bb} ({ckpt.parent.name}) hier={h} | "
                f"VAL KL3={vm['per_class_f1']['KL3']:.4f} KL4={vm['per_class_f1']['KL4']:.4f} | "
                f"TEST KL3={tm['per_class_f1']['KL3']:.4f} KL4={tm['per_class_f1']['KL4']:.4f} Acc={tm['accuracy']:.4f}"
            )

    # Equal-weight ensemble first
    eq_vp = sum(val_probs) / len(val_probs)
    eq_tp = sum(test_probs) / len(test_probs)
    equal_ensemble_metrics: dict[str, dict[str, object]] = {}
    for h in (False, True):
        ev = compute_classification_metrics(y_val, decode_predictions_from_probs(torch.from_numpy(eq_vp), h).numpy(), eq_vp)
        et = compute_classification_metrics(y_test, decode_predictions_from_probs(torch.from_numpy(eq_tp), h).numpy(), eq_tp)
        key = "hier_true" if h else "hier_false"
        equal_ensemble_metrics[key] = {"val_metrics": ev, "test_metrics": et}
        print(
            f"Equal Ensemble (n={len(specs)}) hier={h} | "
            f"VAL KL3={ev['per_class_f1']['KL3']:.4f} KL4={ev['per_class_f1']['KL4']:.4f} | "
            f"TEST KL3={et['per_class_f1']['KL3']:.4f} KL4={et['per_class_f1']['KL4']:.4f} "
            f"Macro={et['macro_f1']:.4f} Acc={et['accuracy']:.4f} "
            f"BinSens={et['binary_oa_sensitivity']:.4f} BinAUC={et['binary_oa_roc_auc']:.4f}"
        )

    # Strict val-only weight selection across discrete simplex weights
    best_score = -1.0
    best_w = None
    best_h = False
    grid = [0, 1, 2, 3, 4]
    for raw_w in product(grid, repeat=len(specs)):
        if sum(raw_w) == 0:
            continue
        w = [x / sum(raw_w) for x in raw_w]
        vp = sum(wi * p for wi, p in zip(w, val_probs))
        for h in (False, True):
            v_pred = decode_predictions_from_probs(torch.from_numpy(vp), h).numpy()
            vm = compute_classification_metrics(y_val, v_pred, vp)
            sc = compute_selection_score(vm, "oa_balanced")
            if sc > best_score:
                best_score = sc
                best_w = w
                best_h = h

    assert best_w is not None
    sel_vp = sum(wi * p for wi, p in zip(best_w, val_probs))
    sel_tp = sum(wi * p for wi, p in zip(best_w, test_probs))
    sel_vm = compute_classification_metrics(
        y_val, decode_predictions_from_probs(torch.from_numpy(sel_vp), best_h).numpy(), sel_vp
    )
    sel_tm = compute_classification_metrics(
        y_test, decode_predictions_from_probs(torch.from_numpy(sel_tp), best_h).numpy(), sel_tp
    )
    out = {
        "models": [f"{bb}:{ckpt}" for bb, ckpt in specs],
        "equal_ensemble": equal_ensemble_metrics,
        "val_selected_weights": [round(x, 4) for x in best_w],
        "val_selected_hierarchical_oa": best_h,
        "val_metrics": sel_vm,
        "test_metrics": sel_tm,
    }
    if args.output_json:
        out_path = resolve_project_path(args.output_json, _ROOT)
        ensure_dir(out_path.parent)
        save_json(out, out_path)
    print("\n=== VAL-SELECTED ENSEMBLE RESULT ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
