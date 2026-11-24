"""Shared validation metrics (KID, FID, SSIM) for CycleGAN and Diffusion without test leakage."""

import math
from typing import Any

import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device: torch.device | None = None) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g /= g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d.unsqueeze(0).unsqueeze(0)


def _to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.float().clamp(-1.0, 1.0) + 1.0) * 0.5


@torch.no_grad()
def compute_batch_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> float:
    """Compute mean SSIM over a batch of [-1, 1] tensors (B, C, H, W)."""
    if x.shape != y.shape:
        raise ValueError(f"SSIM inputs must share shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
    x01 = _to_unit_range(x)
    y01 = _to_unit_range(y)

    channels = x01.shape[1]
    win = _gaussian_window(window_size=window_size, device=x01.device).repeat(channels, 1, 1, 1)
    pad = window_size // 2

    mu_x = F.conv2d(x01, win, padding=pad, groups=channels)
    mu_y = F.conv2d(y01, win, padding=pad, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x01 * x01, win, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y01 * y01, win, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x01 * y01, win, padding=pad, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / den.clamp(min=1e-12)
    return float(ssim_map.mean().item())


class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 pool3 (2048-d) feature extractor for FID and unbiased KID."""

    def __init__(self, device: torch.device, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        else:
            model = inception_v3(weights=None, aux_logits=False, init_weights=False)
        model.fc = nn.Identity()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model.to(device)
        self.device = device
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> np.ndarray:
        """Extract (N, 2048) features from [-1, 1] images."""
        x = images.to(self.device).float()
        x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = self.model(x)
        if hasattr(feats, "logits"):
            feats = feats.logits
        return feats.detach().cpu().numpy()


def compute_kid_from_features(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """Compute unbiased Kernel Inception Distance (polynomial kernel MMD^2, Binkowski et al. 2018)."""
    real_f = np.asarray(real_feats, dtype=np.float64)
    fake_f = np.asarray(fake_feats, dtype=np.float64)
    m, d = real_f.shape
    n, d_f = fake_f.shape
    if d != d_f or m < 2 or n < 2:
        raise ValueError(f"Need >=2 samples per domain with matching dims, got {real_f.shape} & {fake_f.shape}")

    k_rr = (real_f @ real_f.T / float(d) + 1.0) ** 3
    k_ff = (fake_f @ fake_f.T / float(d) + 1.0) ** 3
    k_rf = (real_f @ fake_f.T / float(d) + 1.0) ** 3

    sum_rr = (k_rr.sum() - np.trace(k_rr)) / float(m * (m - 1))
    sum_ff = (k_ff.sum() - np.trace(k_ff)) / float(n * (n - 1))
    sum_rf = k_rf.mean()
    return float(sum_rr + sum_ff - 2.0 * sum_rf)


def compute_fid_from_features(real_feats: np.ndarray, fake_feats: np.ndarray, eps: float = 1e-6) -> float:
    """Compute Frechet Inception Distance with diagonal regularization for small val sets."""
    real_f = np.asarray(real_feats, dtype=np.float64)
    fake_f = np.asarray(fake_feats, dtype=np.float64)
    mu_r = np.mean(real_f, axis=0)
    mu_f = np.mean(fake_f, axis=0)
    sigma_r = np.cov(real_f, rowvar=False) + np.eye(real_f.shape[1]) * eps
    sigma_f = np.cov(fake_f, rowvar=False) + np.eye(fake_f.shape[1]) * eps

    diff = mu_r - mu_f
    covmean = linalg.sqrtm(sigma_r @ sigma_f)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_f) - 2.0 * np.trace(covmean))
    return max(0.0, fid)


@torch.no_grad()
def extract_features_from_loader(
    extractor: InceptionFeatureExtractor,
    loader: DataLoader,
    image_key: str = "image",
) -> np.ndarray:
    """Extract Inception features over all images in a validation DataLoader."""
    chunks: list[np.ndarray] = []
    for batch in loader:
        imgs = batch[image_key] if isinstance(batch, dict) else batch[0]
        chunks.append(extractor(imgs))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def evaluate_translation_validation(
    forward_fn,
    backward_fn,
    val_source_loader: DataLoader,
    real_target_feats: np.ndarray,
    extractor: InceptionFeatureExtractor,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate anatomical preservation (SSIM/L1) and target-domain realism (KID/FID) on val."""
    fake_chunks: list[np.ndarray] = []
    total_samples = 0
    cycle_l1_sum = 0.0
    cycle_ssim_sum = 0.0
    delta_l1_sum = 0.0
    struct_ssim_sum = 0.0

    for batch in val_source_loader:
        real_a = batch["image"].to(device, non_blocking=True)
        bsz = real_a.shape[0]
        fake_b = forward_fn(real_a)
        rec_a = backward_fn(fake_b)

        cycle_l1_sum += float(F.l1_loss(rec_a, real_a, reduction="mean").item()) * bsz
        delta_l1_sum += float(F.l1_loss(fake_b, real_a, reduction="mean").item()) * bsz
        cycle_ssim_sum += compute_batch_ssim(real_a, rec_a) * bsz
        struct_ssim_sum += compute_batch_ssim(real_a, fake_b) * bsz
        fake_chunks.append(extractor(fake_b))
        total_samples += bsz

    fake_feats = np.concatenate(fake_chunks, axis=0)
    val_kid = compute_kid_from_features(real_target_feats, fake_feats)
    val_fid = compute_fid_from_features(real_target_feats, fake_feats)

    denom = max(1, total_samples)
    metrics = {
        "val_samples": int(total_samples),
        "val_target_samples": int(real_target_feats.shape[0]),
        "val_kid": val_kid,
        "val_fid": val_fid,
        "val_cycle_ssim": cycle_ssim_sum / denom,
        "val_cycle_l1": cycle_l1_sum / denom,
        "val_delta_l1": delta_l1_sum / denom,
        "val_struct_ssim": struct_ssim_sum / denom,
    }
    for key, val in metrics.items():
        if isinstance(val, float) and not math.isfinite(val):
            raise ValueError(f"Non-finite validation metric '{key}': {val}")
    return metrics
