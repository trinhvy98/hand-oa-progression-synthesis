"""Shared PyTorch datasets with split guards against test-set leakage."""

import io
from pathlib import Path
import random
from typing import Callable, Iterable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from hand_oa.data.split import verify_no_data_leakage


def _filter_manifest(
    manifest_df: pd.DataFrame,
    split: str,
    joints: Iterable[str] | None = None,
    grades: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Validate manifest integrity and filter by split, joints, and KL grades."""
    verify_no_data_leakage(manifest_df)
    subset = manifest_df[manifest_df["split"] == split].copy()
    if joints is not None:
        joint_set = {j.upper() for j in joints}
        subset = subset[subset["joint"].str.upper().isin(joint_set)]
    if grades is not None:
        grade_set = {int(g) for g in grades}
        subset = subset[subset["kl_grade"].isin(grade_set)]
    return subset.reset_index(drop=True)


class UnpairedDomainDataset(Dataset):
    """Unpaired A<->B dataset for CycleGAN or unpaired Diffusion translation."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        images_dir: str | Path,
        split: str = "train",
        domain_a_grades: Iterable[int] = (0,),
        domain_b_grades: Iterable[int] = (3,),
        joints: Iterable[str] | None = None,
        transform: Callable | None = None,
        is_train: bool = True,
        allow_test: bool = False,
    ) -> None:
        if split == "test" and not allow_test:
            raise ValueError(
                "Refusing to load 'test' split into UnpairedDomainDataset. "
                "The locked test set must never be used for generative training or tuning."
            )
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_train = is_train

        df_a = _filter_manifest(manifest_df, split=split, joints=joints, grades=domain_a_grades)
        df_b = _filter_manifest(manifest_df, split=split, joints=joints, grades=domain_b_grades)
        if df_a.empty or df_b.empty:
            raise ValueError(
                f"Empty domain for split='{split}': Domain A={len(df_a)}, Domain B={len(df_b)}"
            )

        self.records_a = df_a[["filename", "patient_id", "kl_grade"]].to_dict("records")
        self.records_b = df_b[["filename", "patient_id", "kl_grade"]].to_dict("records")

    def __len__(self) -> int:
        return max(len(self.records_a), len(self.records_b))

    def _load_image(self, filename: str):
        path = self.images_dir / filename
        with Image.open(path) as img:
            loaded = img.copy()
        return self.transform(loaded) if self.transform is not None else loaded

    def __getitem__(self, index: int) -> dict[str, object]:
        rec_a = self.records_a[index % len(self.records_a)]
        if self.is_train:
            idx_b = random.randint(0, len(self.records_b) - 1)
        else:
            idx_b = index % len(self.records_b)
        rec_b = self.records_b[idx_b]

        return {
            "A": self._load_image(rec_a["filename"]),
            "B": self._load_image(rec_b["filename"]),
            "A_filename": rec_a["filename"],
            "B_filename": rec_b["filename"],
            "A_patient_id": rec_a["patient_id"],
            "B_patient_id": rec_b["patient_id"],
        }


class SingleDomainDataset(Dataset):
    """Single-domain dataset for synthetic generation from train/val pools only."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        images_dir: str | Path,
        split: str = "val",
        grades: Iterable[int] = (0,),
        joints: Iterable[str] | None = None,
        transform: Callable | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        if split == "test":
            raise ValueError(
                "Refusing to synthesize augmentation images from the 'test' split. "
                "Only 'train' or 'val' may serve as generative source pools."
            )
        self.images_dir = Path(images_dir)
        self.transform = transform

        df = _filter_manifest(manifest_df, split=split, joints=joints, grades=grades)
        if df.empty:
            raise ValueError(f"No samples found for split='{split}', grades={list(grades)}")
        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        self.records = df[["sample_id", "patient_id", "filename", "kl_grade"]].to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        rec = self.records[index]
        with Image.open(self.images_dir / rec["filename"]) as img:
            loaded = img.copy()
        tensor = self.transform(loaded) if self.transform is not None else loaded
        return {
            "image": tensor,
            "sample_id": rec["sample_id"],
            "patient_id": rec["patient_id"],
            "filename": rec["filename"],
            "kl_grade": int(rec["kl_grade"]),
        }


class JointClassificationDataset(Dataset):
    """KL-grade classification dataset supporting optional synthetic data on train split only."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        images_dir: str | Path,
        split: str,
        joints: Iterable[str] | None = None,
        transform: Callable | None = None,
        synthetic_items: list[tuple[Path, int]] | None = None,
        oversample_multipliers: dict[int, int] | None = None,
        synthetic_oversample_multipliers: dict[int, int] | None = None,
        cache_in_memory: bool = True,
        pre_resize_size: int | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.split = split

        if synthetic_items and split != "train":
            raise ValueError(
                f"Synthetic augmentation is only permitted on split='train', got split='{split}'"
            )
        if (oversample_multipliers or synthetic_oversample_multipliers) and split != "train":
            raise ValueError(
                f"Oversampling is only permitted on split='train', got split='{split}'"
            )

        df = _filter_manifest(manifest_df, split=split, joints=joints)
        if df.empty:
            raise ValueError(f"No samples found for split='{split}'")

        samples: list[tuple[Path, int, bool]] = []
        sample_aux: list[tuple[float, float, float, float]] = []
        for row in df.to_dict("records"):
            samples.append((self.images_dir / row["filename"], int(row["kl_grade"]), False))
            if split == "train" and all(k in row for k in ("op_score", "jsn_score", "er_score")):
                op_v, jsn_v, er_v = row["op_score"], row["jsn_score"], row["er_score"]
                if pd.notna(op_v) and pd.notna(jsn_v) and pd.notna(er_v):
                    sample_aux.append(
                        (
                            float(min(max(float(op_v) / 3.0, 0.0), 1.0)),
                            float(min(max(float(jsn_v) / 3.0, 0.0), 1.0)),
                            float(min(max(float(er_v), 0.0), 1.0)),
                            1.0,
                        )
                    )
                    continue
            sample_aux.append((0.0, 0.0, 0.0, 0.0))

        if synthetic_items:
            for syn_path, syn_label in synthetic_items:
                samples.append((Path(syn_path), int(syn_label), True))
                sample_aux.append((0.0, 0.0, 0.0, 0.0))

        if oversample_multipliers or synthetic_oversample_multipliers:
            expanded: list[tuple[Path, int, bool]] = []
            expanded_aux: list[tuple[float, float, float, float]] = []
            for item, aux in zip(samples, sample_aux):
                mult_map = (
                    synthetic_oversample_multipliers
                    if (item[2] and synthetic_oversample_multipliers is not None)
                    else oversample_multipliers
                )
                mult = int(mult_map.get(item[1], 1)) if mult_map else 1
                rep = max(1, mult)
                expanded.extend([item] * rep)
                expanded_aux.extend([aux] * rep)
            samples = expanded
            sample_aux = expanded_aux

        self.samples = samples
        self.sample_aux = sample_aux
        self._png_cache: dict[Path, bytes] = {}
        self._resized_cache: dict[Path, Image.Image] = {}
        unique_paths = {item[0] for item in self.samples}
        if pre_resize_size is not None:
            size = (int(pre_resize_size), int(pre_resize_size))
            for p in unique_paths:
                with Image.open(p) as img:
                    self._resized_cache[p] = img.convert("RGB").resize(
                        size, Image.Resampling.BICUBIC
                    )
        elif cache_in_memory:
            self._png_cache = {p: p.read_bytes() for p in unique_paths}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        path, label, _is_synthetic = self.samples[index]
        resized = self._resized_cache.get(path)
        if resized is not None:
            loaded = resized.copy()
        else:
            raw = self._png_cache.get(path)
            if raw is not None:
                with Image.open(io.BytesIO(raw)) as img:
                    loaded = img.copy()
            else:
                with Image.open(path) as img:
                    loaded = img.copy()
        tensor = self.transform(loaded) if self.transform is not None else loaded
        return tensor, label
