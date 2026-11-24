"""High-throughput GPU/RAM preloaded batch loader with vectorized CUDA augmentation."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from hand_oa.data.datasets import JointClassificationDataset

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _SampleCountDataset:
    """Lightweight dataset view exposing sample count for logging and reports."""

    def __init__(self, num_samples: int) -> None:
        self._num_samples = int(num_samples)

    def __len__(self) -> int:
        return self._num_samples


class FastPreloadedBatchLoader:
    """In-memory / VRAM image batch iterator eliminating per-epoch CPU PIL bottlenecks.

    Applies ImageNet normalization and (when `is_train=True`) batched horizontal flip
    and random affine perturbations directly on the target compute device.
    """

    def __init__(
        self,
        dataset: JointClassificationDataset,
        image_size: int,
        batch_size: int,
        device: torch.device,
        is_train: bool = True,
        max_rotate_deg: float = 10.0,
        max_translate_frac: float = 0.05,
        max_translate_y_frac: float | None = None,
        max_scale_jitter: float = 0.07,
        brightness_jitter: float = 0.10,
        contrast_jitter: float = 0.12,
        cutout_prob: float = 0.0,
        return_aux: bool = False,
    ) -> None:
        self.dataset = _SampleCountDataset(len(dataset))
        self.batch_size = int(batch_size)
        self.device = device
        self.is_train = bool(is_train)
        self.max_rotate_rad = float(np.deg2rad(max_rotate_deg))
        self.max_translate_x_norm = 2.0 * float(max_translate_frac)
        ty_frac = max_translate_frac if max_translate_y_frac is None else max_translate_y_frac
        self.max_translate_y_norm = 2.0 * float(ty_frac)
        self.max_scale_jitter = float(max_scale_jitter)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.cutout_prob = float(cutout_prob)
        self.return_aux = bool(return_aux and is_train)

        unique_paths: list[Path] = []
        path_to_idx: dict[Path, int] = {}
        sample_indices: list[int] = []
        sample_labels: list[int] = []

        for path, label, _is_syn in dataset.samples:
            idx = path_to_idx.get(path)
            if idx is None:
                idx = len(unique_paths)
                path_to_idx[path] = idx
                unique_paths.append(path)
            sample_indices.append(idx)
            sample_labels.append(int(label))

        size = (int(image_size), int(image_size))

        def _decode_one(path: Path) -> tuple[np.ndarray, bool]:
            with Image.open(path) as img:
                rgb = np.asarray(
                    img.convert("RGB").resize(size, Image.Resampling.BICUBIC),
                    dtype=np.uint8,
                )
            is_gray = bool(
                np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
                and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
            )
            return (rgb[:, :, 0:1] if is_gray else rgb), is_gray

        with ThreadPoolExecutor(max_workers=8) as pool:
            loaded_pairs = list(pool.map(_decode_one, unique_paths))

        all_grayscale = all(flag for _, flag in loaded_pairs)
        if all_grayscale:
            # Store 1-channel uint8 (3x less VRAM) and broadcast to 3 channels on-the-fly
            arr = np.stack([img for img, _ in loaded_pairs], axis=0)
        else:
            arr = np.stack(
                [np.repeat(img, 3, axis=2) if flag else img for img, flag in loaded_pairs],
                axis=0,
            )

        self.unique_images = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
        self.sample_indices = torch.tensor(sample_indices, dtype=torch.long, device=device)
        self.labels = torch.tensor(sample_labels, dtype=torch.long, device=device)
        raw_aux = getattr(dataset, "sample_aux", None)
        if raw_aux is not None and len(raw_aux) == len(sample_indices):
            self.aux_targets = torch.tensor(raw_aux, dtype=torch.float32, device=device)
        else:
            self.aux_targets = torch.zeros((len(sample_indices), 4), dtype=torch.float32, device=device)
        self.mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return (len(self.sample_indices) + self.batch_size - 1) // self.batch_size

    def _augment_batch(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        flip_mask = torch.rand(b, 1, 1, 1, device=self.device) < 0.5
        x = torch.where(flip_mask, torch.flip(x, dims=[-1]), x)

        angles = (torch.rand(b, device=self.device) * 2.0 - 1.0) * self.max_rotate_rad
        tx = (torch.rand(b, device=self.device) * 2.0 - 1.0) * self.max_translate_x_norm
        ty = (torch.rand(b, device=self.device) * 2.0 - 1.0) * self.max_translate_y_norm
        if self.max_scale_jitter > 0.0:
            scales = 1.0 + (torch.rand(b, device=self.device) * 2.0 - 1.0) * self.max_scale_jitter
            inv_scale = 1.0 / scales
        else:
            inv_scale = torch.ones(b, device=self.device)

        cos_a = torch.cos(angles) * inv_scale
        sin_a = torch.sin(angles) * inv_scale
        theta = torch.zeros((b, 2, 3), dtype=x.dtype, device=self.device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = sin_a
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = -sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        if self.contrast_jitter > 0.0 or self.brightness_jitter > 0.0:
            mean_val = x.mean(dim=(-2, -1), keepdim=True)
            c_factor = 1.0 + (torch.rand(b, 1, 1, 1, device=self.device) * 2.0 - 1.0) * self.contrast_jitter
            b_shift = (torch.rand(b, 1, 1, 1, device=self.device) * 2.0 - 1.0) * self.brightness_jitter
            x = ((x - mean_val) * c_factor + mean_val + b_shift).clamp_(0.0, 1.0)
        if self.cutout_prob > 0.0:
            _, _, h, w = x.shape
            apply_c = torch.rand(b, 1, 1, 1, device=self.device) < self.cutout_prob
            yy = torch.arange(h, device=self.device).view(1, 1, h, 1)
            xx = torch.arange(w, device=self.device).view(1, 1, 1, w)
            cy = torch.randint(int(0.15 * h), int(0.85 * h) + 1, (b, 1, 1, 1), device=self.device)
            cx = torch.randint(int(0.15 * w), int(0.85 * w) + 1, (b, 1, 1, 1), device=self.device)
            rh = max(2, int(0.10 * h))
            rw = max(2, int(0.10 * w))
            box = (yy >= (cy - rh)) & (yy <= (cy + rh)) & (xx >= (cx - rw)) & (xx <= (cx + rw))
            x = torch.where(apply_c & box, torch.zeros((), device=self.device, dtype=x.dtype), x)
        return x

    def __iter__(self):
        n = len(self.sample_indices)
        perm = torch.randperm(n, device=self.device) if self.is_train else None

        for start in range(0, n, self.batch_size):
            order = perm[start : start + self.batch_size] if perm is not None else slice(start, start + self.batch_size)
            img_idx = self.sample_indices[order]
            targets = self.labels[order]

            x = self.unique_images[img_idx].float().div_(255.0)
            if self.is_train:
                x = self._augment_batch(x)
            if x.size(1) == 1:
                x = x.expand(-1, 3, -1, -1)
            x = (x - self.mean) / self.std
            if self.return_aux:
                yield x, targets, self.aux_targets[order]
            else:
                yield x, targets


def build_fast_classification_loaders(
    manifest: pd.DataFrame,
    images_dir: str | Path,
    joints: Iterable[str],
    image_size: int,
    batch_size: int,
    device: torch.device,
    synthetic_items: list[tuple[Path, int]] | None = None,
    oversample: dict[int, int] | None = None,
    synthetic_oversample: dict[int, int] | None = None,
    include_test: bool = True,
    max_translate_frac: float = 0.05,
    max_translate_y_frac: float | None = None,
    max_scale_jitter: float = 0.07,
    brightness_jitter: float = 0.10,
    contrast_jitter: float = 0.12,
    cutout_prob: float = 0.0,
    return_aux: bool = False,
) -> tuple[FastPreloadedBatchLoader, FastPreloadedBatchLoader, FastPreloadedBatchLoader | None]:
    """Build GPU/RAM-preloaded loaders preserving strict split and leakage guards."""
    train_ds = JointClassificationDataset(
        manifest_df=manifest,
        images_dir=images_dir,
        split="train",
        joints=joints,
        transform=None,
        synthetic_items=synthetic_items,
        oversample_multipliers=oversample,
        synthetic_oversample_multipliers=synthetic_oversample,
        cache_in_memory=False,
    )
    val_ds = JointClassificationDataset(
        manifest_df=manifest,
        images_dir=images_dir,
        split="val",
        joints=joints,
        transform=None,
        cache_in_memory=False,
    )
    train_loader = FastPreloadedBatchLoader(
        train_ds,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        is_train=True,
        max_translate_frac=max_translate_frac,
        max_translate_y_frac=max_translate_y_frac,
        max_scale_jitter=max_scale_jitter,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        cutout_prob=cutout_prob,
        return_aux=return_aux,
    )
    val_loader = FastPreloadedBatchLoader(
        val_ds, image_size=image_size, batch_size=batch_size, device=device, is_train=False
    )

    test_loader = None
    if include_test:
        test_ds = JointClassificationDataset(
            manifest_df=manifest,
            images_dir=images_dir,
            split="test",
            joints=joints,
            transform=None,
            cache_in_memory=False,
        )
        test_loader = FastPreloadedBatchLoader(
            test_ds, image_size=image_size, batch_size=batch_size, device=device, is_train=False
        )
    return train_loader, val_loader, test_loader
