"""Smoke tests for CycleGAN architectures, training step, checkpointing, and generation."""

import os
from pathlib import Path
import socket

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cyclegan.engine import CycleGANTrainer, generate_synthetic_images
from cyclegan.models import CycleGANModel, PatchGANDiscriminator, ResNetGenerator
from hand_oa.data import assign_patient_splits
from hand_oa.evaluation import (
    compute_batch_ssim,
    compute_fid_from_features,
    compute_kid_from_features,
)


def test_cyclegan_network_shapes() -> None:
    gen = ResNetGenerator(in_channels=1, out_channels=1, ngf=16, n_res_blocks=2)
    disc = PatchGANDiscriminator(in_channels=1, ndf=16)

    x = torch.randn(2, 1, 128, 128)
    fake = gen(x)
    assert fake.shape == (2, 1, 128, 128)
    assert float(fake.detach().min()) >= -1.0 and float(fake.detach().max()) <= 1.0

    pred = disc(fake)
    assert pred.ndim == 4 and pred.shape[0] == 2 and pred.shape[1] == 1


def test_cyclegan_train_step_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = CycleGANModel(
        in_channels=1,
        ngf=16,
        ndf=16,
        n_res_blocks=2,
        pool_size=4,
        epochs=4,
        decay_start_epoch=2,
        device=torch.device("cpu"),
    )
    real_a = torch.randn(2, 1, 64, 64)
    real_b = torch.randn(2, 1, 64, 64)

    losses = model.train_step(real_a, real_b)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())

    ckpt_path = model.save_checkpoint(tmp_path / "epoch_0001.pt", epoch=1)
    reloaded = CycleGANModel(
        in_channels=1,
        ngf=16,
        ndf=16,
        n_res_blocks=2,
        pool_size=4,
        epochs=4,
        decay_start_epoch=2,
        device=torch.device("cpu"),
    )
    loaded_epoch = reloaded.load_checkpoint(ckpt_path, load_optimizers=True)
    assert loaded_epoch == 1

    out_orig = model.translate(real_a, direction="a2b")
    out_reloaded = reloaded.translate(real_a, direction="a2b")
    assert torch.allclose(out_orig, out_reloaded, atol=1e-6)


def test_trainer_and_generator_end_to_end(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(18):
        pid = f"800{idx:04d}"
        kl = 0 if idx % 2 == 0 else 3
        fname = f"{pid}_dip2.png"
        Image.new("L", (64, 64), color=100 + kl * 20).save(images_dir / fname)
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
    manifest = assign_patient_splits(
        pd.DataFrame(rows), train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    manifest_csv = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    dummy_xlsx = tmp_path / "dummy.xlsx"
    dummy_xlsx.write_bytes(b"unused-when-manifest-exists")

    data_cfg = {
        "paths": {
            "excel_path": str(dummy_xlsx),
            "images_dir": str(images_dir),
            "manifest_path": str(manifest_csv),
        },
        "dataset": {
            "joints": ["DIP2"],
            "visit": "v00",
            "valid_kl_grades": [0, 1, 2, 3, 4],
        },
        "split": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "seed": 42},
    }
    out_dir = tmp_path / "out_cg"
    cg_cfg = {
        "experiment": {"name": "smoke", "output_dir": str(out_dir), "seed": 42},
        "data": {
            "domain_a_grades": [0],
            "domain_b_grades": [3],
            "joints": ["DIP2"],
            "image_size": 64,
            "in_channels": 1,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "ngf": 16,
            "ndf": 16,
            "n_res_blocks": 2,
            "gan_mode": "lsgan",
            "pool_size": 4,
        },
        "training": {
            "epochs": 1,
            "decay_start_epoch": 1,
            "lr": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
            "lambda_cycle_a": 10.0,
            "lambda_cycle_b": 10.0,
            "lambda_identity": 0.5,
            "save_every_epochs": 1,
            "sample_every_epochs": 1,
        },
        "generation": {
            "source_split": "train",
            "direction": "a2b",
            "target_label": 3,
            "max_samples": 2,
            "output_subdir": "syn_kl3",
        },
    }

    trainer = CycleGANTrainer(
        data_cfg=data_cfg,
        cyclegan_cfg=cg_cfg,
        project_root=tmp_path,
        device=torch.device("cpu"),
    )
    history = trainer.fit(max_epochs=1, max_steps_per_epoch=2)
    assert len(history) == 1
    assert (out_dir / "checkpoints" / "epoch_0001.pt").is_file()
    assert (out_dir / "samples" / "epoch_0001.png").is_file()

    # Verify resume_checkpoint advances seamlessly from epoch 1 to epoch 2
    resumed = trainer.fit(
        max_epochs=2,
        max_steps_per_epoch=1,
        resume_checkpoint=out_dir / "checkpoints" / "epoch_0001.pt",
    )
    assert [row["epoch"] for row in resumed] == [1, 2]
    assert (out_dir / "checkpoints" / "epoch_0002.pt").is_file()

    syn_manifest = generate_synthetic_images(
        data_cfg=data_cfg,
        cyclegan_cfg=cg_cfg,
        checkpoint_path=out_dir / "checkpoints" / "epoch_0002.pt",
        project_root=tmp_path,
        device=torch.device("cpu"),
    )
    syn_df = pd.read_csv(syn_manifest)
    assert len(syn_df) == 2
    assert (syn_df["source_split"] == "train").all()
    assert all(Path(p).is_file() for p in syn_df["image_path"])


def _ddp_worker(rank: int, world_size: int, port: int, out_dir_str: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        model = CycleGANModel(
            in_channels=1,
            ngf=16,
            ndf=16,
            n_res_blocks=2,
            pool_size=4,
            epochs=2,
            decay_start_epoch=1,
            is_distributed=True,
            local_rank=rank,
            device=torch.device("cpu"),
        )
        torch.manual_seed(100 + rank)
        losses = model.train_step(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64))
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
        if rank == 0:
            model.save_checkpoint(Path(out_dir_str) / "ddp_ckpt.pt", epoch=1)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_ddp_two_ranks_gloo_smoke(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    mp.spawn(_ddp_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    assert (tmp_path / "ddp_ckpt.pt").is_file()


def test_generative_metrics_and_early_stopping(tmp_path: Path) -> None:
    img = torch.rand(2, 1, 64, 64) * 2.0 - 1.0
    assert abs(compute_batch_ssim(img, img) - 1.0) < 1e-4

    rng = np.random.default_rng(42)
    feats_a = rng.normal(0.0, 1.0, size=(12, 32))
    feats_b = rng.normal(2.0, 1.0, size=(12, 32))
    kid_same = compute_kid_from_features(feats_a, feats_a + rng.normal(0, 0.01, feats_a.shape))
    kid_diff = compute_kid_from_features(feats_a, feats_b)
    assert kid_diff > kid_same
    assert compute_fid_from_features(feats_a, feats_b) > 0.0

    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(20):
        pid = f"700{idx:04d}"
        kl = 0 if idx % 2 == 0 else 3
        fname = f"{pid}_dip2.png"
        Image.new("L", (64, 64), color=90 + kl * 30).save(images_dir / fname)
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
    manifest = assign_patient_splits(
        pd.DataFrame(rows), train_ratio=0.5, val_ratio=0.3, test_ratio=0.2, seed=42
    )
    manifest_csv = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    dummy_xlsx = tmp_path / "dummy.xlsx"
    dummy_xlsx.write_bytes(b"unused")

    out_dir = tmp_path / "out_es"
    data_cfg = {
        "paths": {"excel_path": str(dummy_xlsx), "images_dir": str(images_dir), "manifest_path": str(manifest_csv)},
        "dataset": {"joints": ["DIP2"], "visit": "v00", "valid_kl_grades": [0, 1, 2, 3, 4]},
        "split": {"train_ratio": 0.5, "val_ratio": 0.3, "test_ratio": 0.2, "seed": 42},
    }
    cg_cfg = {
        "experiment": {"name": "es_smoke", "output_dir": str(out_dir), "seed": 42},
        "data": {"domain_a_grades": [0], "domain_b_grades": [3], "joints": ["DIP2"], "image_size": 64, "in_channels": 1, "batch_size": 2, "num_workers": 0},
        "model": {"ngf": 8, "ndf": 8, "n_res_blocks": 1, "gan_mode": "lsgan", "pool_size": 4},
        "training": {"epochs": 10, "decay_start_epoch": 5, "lr": 0.0002, "beta1": 0.5, "beta2": 0.999, "lambda_cycle_a": 10.0, "lambda_cycle_b": 10.0, "lambda_identity": 0.0, "save_every_epochs": 1, "sample_every_epochs": 1, "use_amp": False},
        "early_stopping": {
            "enabled": True,
            "eval_every_epochs": 1,
            "min_epochs": 1,
            "patience": 1,
            "min_delta": 1000.0,  # Force patience exhaustion after first non-improving check
            "min_cycle_ssim": -1.0,
            "min_delta_l1": 0.0,
            "use_pretrained_inception": False,
        },
        "generation": {"source_split": "train", "direction": "a2b", "target_label": 3, "max_samples": 2, "output_subdir": "syn"},
    }
    trainer = CycleGANTrainer(data_cfg=data_cfg, cyclegan_cfg=cg_cfg, project_root=tmp_path, device=torch.device("cpu"))
    history = trainer.fit(max_epochs=10, max_steps_per_epoch=1)
    assert len(history) == 2
    assert (out_dir / "val_metrics.csv").is_file()
    assert (out_dir / "checkpoints" / "best_checkpoint.pt").is_file()
    assert (out_dir / "best_checkpoint_info.json").is_file()
