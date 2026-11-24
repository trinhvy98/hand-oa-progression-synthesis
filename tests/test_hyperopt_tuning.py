"""Tests for Focal Loss class weighting, fast GPU batch loader, and multi-worker Hyperopt."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import torch

from evaluation.run_downstream_eval import build_classification_loaders
from evaluation.tune_focal_hyperopt import run_hyperopt_search
from hand_oa.data import assign_patient_splits
from hand_oa.evaluation import (
    FocalLoss,
    build_classifier,
    build_fast_classification_loaders,
    compute_selection_score,
    train_classifier,
)


def test_focal_loss_accepts_class_weights_and_selection_score() -> None:
    weights = torch.tensor([1.0, 1.0, 1.0, 2.5, 2.0])
    crit = FocalLoss(gamma=2.5, weight=weights)
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 2, 3, 4])
    loss = crit(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)

    fake_val = {
        "macro_f1": 0.60,
        "binary_oa_f1": 0.80,
        "per_class_f1": {"KL0": 0.9, "KL1": 0.3, "KL2": 0.7, "KL3": 0.5, "KL4": 0.6},
    }
    assert compute_selection_score(fake_val, "macro_f1") == 0.60
    expected = 0.40 * 0.60 + 0.25 * 0.50 + 0.25 * 0.60 + 0.10 * 0.80
    assert abs(compute_selection_score(fake_val, "oa_balanced") - expected) < 1e-6
    fake_val["val_loss"] = 0.42
    fake_val["val_balanced_loss"] = 0.55
    assert compute_selection_score(fake_val, "val_loss") == -0.42
    assert compute_selection_score(fake_val, "val_balanced_loss") == -0.55


def test_train_classifier_plateau_scheduler_and_val_balanced_loss(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(20):
        pid = f"700{idx:04d}"
        kl = idx % 5
        fname = f"{pid}_dip2.png"
        Image.new("L", (64, 64), color=40 + kl * 35).save(images_dir / fname)
        rows.append(
            {
                "sample_id": f"{pid}_dip2",
                "patient_id": pid,
                "joint": "DIP2",
                "joint_group": "DIP",
                "hand": 0,
                "kl_grade": kl,
                "filename": fname,
            }
        )
    manifest = assign_patient_splits(pd.DataFrame(rows), seed=42)
    train_loader, val_loader, _ = build_fast_classification_loaders(
        manifest=manifest,
        images_dir=images_dir,
        joints=["DIP2"],
        image_size=64,
        batch_size=8,
        device=torch.device("cpu"),
        include_test=False,
        max_scale_jitter=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    model = build_classifier("resnet50", num_classes=5, pretrained=False, dropout=0.5)
    _, history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        epochs=3,
        lr=1e-3,
        class_weights=[1.0, 1.0, 1.0, 2.0, 2.0],
        patience=5,
        min_best_epoch=1,
        selection_metric="val_balanced_loss",
        scheduler_type="plateau",
        lr_factor=0.5,
        lr_patience=1,
        min_lr=1e-6,
        optimizer_type="adam",
        mixup_alpha=0.2,
        ordinal_weight=0.2,
    )
    assert len(history) == 3
    assert "val_loss" in history[-1]["val"]
    assert "val_balanced_loss" in history[-1]["val"]
    assert history[-1]["val"]["selection_score"] == -history[-1]["val"]["val_balanced_loss"]


def test_hyperopt_search_fast_loader_resume_and_parallel_workers(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(25):
        pid = f"900{idx:04d}"
        kl = idx % 5
        fname = f"{pid}_dip2.png"
        Image.new("L", (64, 64), color=40 + kl * 35).save(images_dir / fname)
        rows.append(
            {
                "sample_id": f"{pid}_dip2",
                "patient_id": pid,
                "joint": "DIP2",
                "joint_group": "DIP",
                "hand": 0,
                "kl_grade": kl,
                "filename": fname,
            }
        )
    manifest = assign_patient_splits(pd.DataFrame(rows), seed=42)
    train_loader, val_loader, test_loader = build_fast_classification_loaders(
        manifest=manifest,
        images_dir=images_dir,
        joints=["DIP2"],
        image_size=64,
        batch_size=8,
        device=torch.device("cpu"),
        include_test=False,
    )
    assert test_loader is None
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape[1:] == (3, 64, 64)
    assert torch.isfinite(batch_x).all()
    assert batch_y.ndim == 1
    sp_model = build_classifier("vgg19_sp", pretrained=False)
    assert sp_model(torch.randn(2, 3, 64, 64)).shape == (2, 5)
    assert isinstance(build_classifier("densenet121", pretrained=False), torch.nn.Module)
    assert isinstance(build_classifier("convnext_tiny", pretrained=False), torch.nn.Module)
    assert isinstance(build_classifier("efficientnet_v2_s", pretrained=False), torch.nn.Module)

    # Verify legacy CPU loader matches shape cleanly
    cpu_train, _, _ = build_classification_loaders(
        manifest=manifest,
        images_dir=images_dir,
        joints=["DIP2"],
        image_size=64,
        batch_size=8,
        num_workers=0,
        include_test=False,
        pre_resize=True,
    )
    cpu_x, _ = next(iter(cpu_train))
    assert cpu_x.shape[1:] == (3, 64, 64)

    out_dir = tmp_path / "hpo_out"
    model_cfg = {
        "backbone": "resnet50",
        "num_classes": 5,
        "pretrained": False,
        "dropout": 0.5,
    }
    train_cfg = {"lr": 1e-3, "weight_decay": 1e-4, "oa_threshold_kl": 3}

    # Initial 2-trial run
    run_hyperopt_search(
        train_loader=train_loader,
        val_loader=val_loader,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=torch.device("cpu"),
        out_dir=out_dir,
        max_trials=2,
        trial_epochs=2,
        patience=2,
        min_best_epoch=1,
        selection_metric="oa_balanced",
        seed=42,
        resume=True,
        worker_id="init",
    )
    df_first = pd.read_csv(out_dir / "hyperopt_trials.csv")
    assert len(df_first) == 2

    # Resume concurrently with 2 parallel workers up to 5 total trials without duplicates
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(
            run_hyperopt_search,
            train_loader,
            val_loader,
            model_cfg,
            train_cfg,
            torch.device("cpu"),
            out_dir,
            5,
            2,
            2,
            1,
            "oa_balanced",
            42,
            True,
            "1",
        )
        f2 = pool.submit(
            run_hyperopt_search,
            train_loader,
            val_loader,
            model_cfg,
            train_cfg,
            torch.device("cpu"),
            out_dir,
            5,
            2,
            2,
            1,
            "oa_balanced",
            42,
            True,
            "2",
        )
        best1 = f1.result()
        best2 = f2.result()

    df_resumed = pd.read_csv(out_dir / "hyperopt_trials.csv")
    assert list(df_resumed["trial_id"]) == [1, 2, 3, 4, 5]
    assert df_resumed.iloc[:2]["focal_gamma"].tolist() == df_first["focal_gamma"].tolist()
    assert len(set(df_resumed["focal_gamma"].tolist())) == 5
    assert json.loads((out_dir / "hyperopt_inflight.json").read_text(encoding="utf-8")) == []

    saved = json.loads((out_dir / "best_focal_config.json").read_text(encoding="utf-8"))
    assert len(saved["class_weights"]) == 5
    assert 1.0 <= saved["focal_gamma"] <= 5.0
    assert 0.5 <= saved["base_alpha"] <= 3.0
    assert saved["best_trial_id"] == best1["best_trial_id"] == best2["best_trial_id"]


def test_split_oversampling_ema_tta_and_aux_subscore_training(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(25):
        pid = f"800{idx:04d}"
        kl = idx % 5
        fname = f"{pid}_dip2.png"
        Image.new("L", (64, 64), color=30 + kl * 40).save(images_dir / fname)
        rows.append(
            {
                "sample_id": f"{pid}_dip2",
                "patient_id": pid,
                "joint": "DIP2",
                "joint_group": "DIP",
                "hand": 0,
                "kl_grade": kl,
                "op_score": min(3, kl),
                "jsn_score": min(3, kl),
                "er_score": 1 if kl >= 4 else 0,
                "filename": fname,
            }
        )
    manifest = assign_patient_splits(pd.DataFrame(rows), seed=42)
    syn_img = tmp_path / "syn_kl4.png"
    Image.new("L", (64, 64), color=210).save(syn_img)

    train_loader, val_loader, test_loader = build_fast_classification_loaders(
        manifest=manifest,
        images_dir=images_dir,
        joints=["DIP2"],
        image_size=64,
        batch_size=8,
        device=torch.device("cpu"),
        synthetic_items=[(syn_img, 4)],
        oversample={3: 2, 4: 4},
        synthetic_oversample={4: 1},
        include_test=True,
        max_translate_frac=0.06,
        max_translate_y_frac=0.14,
        max_scale_jitter=0.14,
        cutout_prob=0.25,
        return_aux=True,
    )
    assert test_loader is not None
    # Real train counts + KL3/KL4 real oversampling minus excessive synthetic duplication
    train_real = manifest[manifest["split"] == "train"]
    expected_len = (
        int((train_real["kl_grade"] < 3).sum())
        + 2 * int((train_real["kl_grade"] == 3).sum())
        + 4 * int((train_real["kl_grade"] == 4).sum())
        + 1
    )
    assert len(train_loader.dataset) == expected_len
    bx, by, baux = next(iter(train_loader))
    assert bx.shape[1:] == (3, 64, 64)
    assert by.ndim == 1
    assert baux.shape[1] == 4

    model = build_classifier("convnext_tiny", num_classes=5, pretrained=False, stochastic_depth_prob=0.2)
    trained, history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        epochs=2,
        lr=1e-3,
        patience=5,
        min_best_epoch=1,
        ema_decay=0.95,
        aux_subscore_weight=0.25,
        oa_bce_weight=0.35,
        severe_pair_weight=0.45,
        tta_views=5,
        hierarchical_oa=True,
        selection_metric="oa_balanced_clean",
    )
    assert len(history) == 2
    from hand_oa.evaluation.classifier import decode_predictions_from_probs, evaluate_classifier

    assert compute_selection_score({"macro_f1": 0.6, "binary_oa_f1": 0.8, "per_class_f1": {"KL3": 0.5, "KL4": 0.6}, "val_balanced_loss": 0.5}, "oa_balanced_clean") < compute_selection_score({"macro_f1": 0.6, "binary_oa_f1": 0.8, "per_class_f1": {"KL3": 0.5, "KL4": 0.6}}, "oa_balanced")

    p_split = torch.tensor([[0.01, 0.02, 0.44, 0.27, 0.26]])
    assert int(decode_predictions_from_probs(p_split, hierarchical_oa=False).item()) == 2
    assert int(decode_predictions_from_probs(p_split, hierarchical_oa=True).item()) == 3
    test_metrics = evaluate_classifier(trained, test_loader, torch.device("cpu"), tta_views=5, hierarchical_oa=True)
    assert "macro_f1" in test_metrics
