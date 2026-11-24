"""Tests verifying patient-level splitting and anti-leakage runtime guards."""

from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from hand_oa.data import (
    JointClassificationDataset,
    SingleDomainDataset,
    UnpairedDomainDataset,
    assign_patient_splits,
    build_manifest_from_excel,
    verify_no_data_leakage,
)
from hand_oa.utils import find_project_root, load_yaml_config, resolve_project_path


def _create_synthetic_fixture(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid_idx in range(20):
        pid = f"900{pid_idx:04d}"
        kl = pid_idx % 5
        for joint in ("DIP2", "DIP3", "DIP4", "DIP5"):
            fname = f"{pid}_{joint.lower()}.png"
            Image.new("L", (64, 64), color=128).save(images_dir / fname)
            rows.append(
                {
                    "sample_id": f"{pid}_{joint.lower()}",
                    "patient_id": pid,
                    "joint": joint,
                    "joint_group": "DIP",
                    "hand": 0,
                    "kl_grade": kl,
                    "filename": fname,
                }
            )
    manifest = assign_patient_splits(pd.DataFrame(rows), seed=42)
    return manifest, images_dir


def test_patient_splits_have_zero_leakage(tmp_path: Path) -> None:
    manifest, _ = _create_synthetic_fixture(tmp_path)
    verify_no_data_leakage(manifest)

    # Inject intentional patient overlap and verify leakage detector catches it
    corrupted = manifest.copy()
    test_pid = corrupted.loc[corrupted["split"] == "test", "patient_id"].iloc[0]
    corrupted.loc[(corrupted["patient_id"] == test_pid) & (corrupted["joint"] == "DIP2"), "split"] = "train"
    with pytest.raises(AssertionError, match="Data leakage detected"):
        verify_no_data_leakage(corrupted)


def test_dataset_runtime_guards_block_test_leakage(tmp_path: Path) -> None:
    manifest, images_dir = _create_synthetic_fixture(tmp_path)

    with pytest.raises(ValueError, match="test"):
        UnpairedDomainDataset(
            manifest_df=manifest,
            images_dir=images_dir,
            split="test",
            domain_a_grades=[0],
            domain_b_grades=[3],
        )

    with pytest.raises(ValueError, match="test"):
        SingleDomainDataset(
            manifest_df=manifest,
            images_dir=images_dir,
            split="test",
            grades=[0],
        )

    dummy_syn = [(images_dir / manifest["filename"].iloc[0], 3)]
    with pytest.raises(ValueError, match="split='train'"):
        JointClassificationDataset(
            manifest_df=manifest,
            images_dir=images_dir,
            split="val",
            synthetic_items=dummy_syn,
        )

    with pytest.raises(ValueError, match="split='train'"):
        JointClassificationDataset(
            manifest_df=manifest,
            images_dir=images_dir,
            split="test",
            oversample_multipliers={3: 5},
        )


def test_real_excel_deduplication_and_manifest_integrity() -> None:
    root = find_project_root()
    data_cfg = load_yaml_config(resolve_project_path("configs/data.yaml", root))
    excel_path = resolve_project_path(data_cfg["paths"]["excel_path"], root)
    images_dir = resolve_project_path(data_cfg["paths"]["images_dir"], root)
    if not excel_path.is_file() or not images_dir.is_dir():
        pytest.skip("Real dataset not present")

    manifest = build_manifest_from_excel(
        excel_path=excel_path,
        images_dir=images_dir,
        joints=data_cfg["dataset"]["joints"],
        visit=data_cfg["dataset"]["visit"],
        valid_kl_grades=data_cfg["dataset"]["valid_kl_grades"],
    )
    assert manifest["sample_id"].is_unique
    split_cfg = data_cfg["split"]
    split_df = assign_patient_splits(
        manifest,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["seed"]),
    )
    verify_no_data_leakage(split_df)
