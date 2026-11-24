"""Loss functions for imbalanced KL-grade classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_per_sample_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Compute unreduced per-sample Cross-Entropy or Focal Loss (without class alpha)."""
    normalized = loss_type.lower()
    if normalized == "ce":
        return F.cross_entropy(logits, targets, reduction="none")
    if normalized == "focal":
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - target_probs).pow(focal_gamma)
        return -focal_factor * target_log_probs
    raise ValueError(f"Unsupported loss_type '{loss_type}'. Expected 'ce' or 'focal'.")


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., ICCV 2017) for imbalanced KL grading."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = compute_per_sample_classification_loss(
            logits=logits,
            targets=targets,
            loss_type="focal",
            focal_gamma=self.gamma,
        )
        if self.weight is not None:
            loss = loss * self.weight.gather(0, targets)
        return loss.mean()


def build_classification_loss(
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Create Cross-Entropy or Focal Loss criterion."""
    normalized = loss_type.lower()
    if normalized == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if normalized == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    raise ValueError(f"Unsupported loss_type '{loss_type}'. Expected 'ce' or 'focal'.")
