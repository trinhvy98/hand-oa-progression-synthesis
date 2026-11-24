"""Patient-grouped stratified train/val/test splitting with strict anti-leakage checks."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from hand_oa.data.metadata import build_manifest_from_excel
from hand_oa.utils.io import ensure_dir, resolve_project_path, save_json

ALLOWED_SPLITS = ("train", "val", "test")


def verify_no_data_leakage(manifest_df: pd.DataFrame) -> None:
    """Assert zero overlap of patients, sample IDs, or filenames across train/val/test."""
    if "split" not in manifest_df.columns:
        raise ValueError("Manifest is missing required 'split' column")

    observed_splits = set(manifest_df["split"].unique())
    invalid = observed_splits - set(ALLOWED_SPLITS)
    if invalid:
        raise ValueError(f"Unexpected split labels detected: {sorted(invalid)}")

    for col in ("patient_id", "sample_id", "filename"):
        if col not in manifest_df.columns:
            continue
        split_sets = {
            split_name: set(manifest_df.loc[manifest_df["split"] == split_name, col].astype(str))
            for split_name in ALLOWED_SPLITS
        }
        for i, split_a in enumerate(ALLOWED_SPLITS):
            for split_b in ALLOWED_SPLITS[i + 1 :]:
                overlap = split_sets[split_a] & split_sets[split_b]
                if overlap:
                    sample_overlap = sorted(overlap)[:5]
                    raise AssertionError(
                        f"Data leakage detected for '{col}' between '{split_a}' and '{split_b}': "
                        f"{len(overlap)} overlapping entries (examples: {sample_overlap})"
                    )


def _safe_stratified_split(
    df: pd.DataFrame,
    test_size: float,
    stratify_col: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified patient split with fallback when sample/class counts are too small."""
    counts = df[stratify_col].value_counts()
    n_classes = len(counts)
    n_right = int(len(df) * test_size)
    n_left = len(df) - n_right
    can_stratify = (
        n_classes > 1
        and bool((counts >= 2).all())
        and n_right >= n_classes
        and n_left >= n_classes
    )
    stratify_vals = df[stratify_col] if can_stratify else None
    left, right = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_vals,
    )
    return left.reset_index(drop=True), right.reset_index(drop=True)


def assign_patient_splits(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.765,
    val_ratio: float = 0.085,
    test_ratio: float = 0.150,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each patient strictly to train, val, or test with zero patient leakage."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio:.6f} "
            f"({train_ratio}, {val_ratio}, {test_ratio})"
        )
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios (train, val, test) must be strictly positive")

    df = manifest_df.copy()
    df["patient_id"] = df["patient_id"].astype(str)

    # Stratify patients by max KL severity and multi-severe-joint bucket so KL3/4 joint counts balance cleanly
    patient_summary = (
        df.groupby("patient_id", as_index=False)
        .agg(
            patient_max_kl=("kl_grade", "max"),
            severe_count=("kl_grade", lambda s: int((s >= 3).sum())),
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    strat_keys = [
        f"{int(mx)}_m" if (int(mx) >= 3 and int(sc) >= 2) else str(int(mx))
        for mx, sc in zip(patient_summary["patient_max_kl"], patient_summary["severe_count"])
    ]
    patient_summary["strat_key"] = strat_keys
    bucket_counts = patient_summary["strat_key"].value_counts()
    patient_summary["strat_key"] = [
        str(int(mx)) if bucket_counts[sk] < 10 else sk
        for mx, sk in zip(patient_summary["patient_max_kl"], patient_summary["strat_key"])
    ]

    train_val_patients, test_patients = _safe_stratified_split(
        patient_summary,
        test_size=test_ratio,
        stratify_col="strat_key",
        seed=seed,
    )
    val_within_trainval = val_ratio / (train_ratio + val_ratio)
    train_patients, val_patients = _safe_stratified_split(
        train_val_patients,
        test_size=val_within_trainval,
        stratify_col="strat_key",
        seed=seed,
    )

    patient_to_split: dict[str, str] = {}
    for pid in train_patients["patient_id"]:
        patient_to_split[pid] = "train"
    for pid in val_patients["patient_id"]:
        patient_to_split[pid] = "val"
    for pid in test_patients["patient_id"]:
        patient_to_split[pid] = "test"

    df["split"] = df["patient_id"].map(patient_to_split)
    verify_no_data_leakage(df)
    return df


def summarize_manifest(manifest_df: pd.DataFrame) -> dict[str, Any]:
    """Produce patient and KL-grade distribution counts per split."""
    verify_no_data_leakage(manifest_df)
    summary: dict[str, Any] = {
        "total_samples": int(len(manifest_df)),
        "total_patients": int(manifest_df["patient_id"].nunique()),
        "splits": {},
    }
    for split_name in ALLOWED_SPLITS:
        sub = manifest_df[manifest_df["split"] == split_name]
        kl_counts = {str(int(k)): int(v) for k, v in sub["kl_grade"].value_counts().sort_index().items()}
        summary["splits"][split_name] = {
            "samples": int(len(sub)),
            "patients": int(sub["patient_id"].nunique()),
            "kl_distribution": kl_counts,
        }
    return summary


def load_or_create_manifest(
    excel_path: str | Path,
    images_dir: str | Path,
    manifest_path: str | Path,
    joints: tuple[str, ...] | list[str] = ("DIP2", "DIP3", "DIP4", "DIP5"),
    visit: str = "v00",
    valid_kl_grades: tuple[int, ...] | list[int] = (0, 1, 2, 3, 4),
    train_ratio: float = 0.765,
    val_ratio: float = 0.085,
    test_ratio: float = 0.150,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Load the frozen split manifest or create and verify it from hand.xlsx."""
    target = Path(manifest_path)
    if target.is_file() and not overwrite:
        df = pd.read_csv(target, dtype={"patient_id": str, "sample_id": str, "filename": str})
        verify_no_data_leakage(df)
        return df

    raw_manifest = build_manifest_from_excel(
        excel_path=excel_path,
        images_dir=images_dir,
        joints=joints,
        visit=visit,
        valid_kl_grades=valid_kl_grades,
    )
    split_manifest = assign_patient_splits(
        raw_manifest,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    ensure_dir(target.parent)
    split_manifest.to_csv(target, index=False)
    save_json(summarize_manifest(split_manifest), target.with_suffix(".summary.json"))
    return split_manifest


def load_manifest_from_config(
    data_cfg: dict[str, Any],
    project_root: str | Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Load or build the frozen patient-level manifest from a parsed data.yaml mapping."""
    paths_cfg = data_cfg["paths"]
    dataset_cfg = data_cfg["dataset"]
    split_cfg = data_cfg["split"]

    images_dir = resolve_project_path(paths_cfg["images_dir"], project_root)
    manifest = load_or_create_manifest(
        excel_path=resolve_project_path(paths_cfg["excel_path"], project_root),
        images_dir=images_dir,
        manifest_path=resolve_project_path(paths_cfg["manifest_path"], project_root),
        joints=dataset_cfg["joints"],
        visit=dataset_cfg["visit"],
        valid_kl_grades=dataset_cfg["valid_kl_grades"],
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["seed"]),
        overwrite=overwrite,
    )
    verify_no_data_leakage(manifest)
    return manifest, images_dir
