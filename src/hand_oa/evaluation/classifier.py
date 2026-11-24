"""Shared downstream KL-grade classifier for evaluating synthetic data utility."""

from copy import deepcopy
import inspect
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from hand_oa.evaluation.losses import (
    build_classification_loss,
    compute_per_sample_classification_loss,
)
from hand_oa.evaluation.metrics import compute_classification_metrics


class SpatialVGG19(nn.Module):
    """VGG19 convolutional features with compact 2x2 avg + 1x1 max spatial head."""

    def __init__(self, features: nn.Module, num_classes: int = 5, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = features
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=max(0.0, dropout * 0.6)),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        avg_f = self.avg_pool(feat).flatten(1)
        max_f = self.max_pool(feat).flatten(1)
        return self.head(torch.cat([avg_f, max_f], dim=1))


def build_classifier(
    backbone: str = "vgg19_bn",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.5,
    stochastic_depth_prob: float = 0.1,
) -> nn.Module:
    """Construct a downstream classification backbone (VGG19, SpatialVGG19, ResNet50, DenseNet121, ConvNeXt)."""
    name = backbone.lower()
    if name == "vgg19_bn":
        weights = models.VGG19_BN_Weights.DEFAULT if pretrained else None
        model = models.vgg19_bn(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-2] = nn.Dropout(p=dropout)
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if name == "vgg19":
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        model = models.vgg19(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-2] = nn.Dropout(p=dropout)
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if name == "vgg19_sp":
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        base = models.vgg19(weights=weights)
        return SpatialVGG19(base.features, num_classes=num_classes, dropout=dropout)
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(
            weights=weights,
            stochastic_depth_prob=float(stochastic_depth_prob),
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model
    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-2] = nn.Dropout(p=dropout)
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported backbone '{backbone}'")


def _shift_scale_batch(x: torch.Tensor, scale: float = 1.0, ty: float = 0.0) -> torch.Tensor:
    """Apply deterministic scale and vertical shift on normalized (B, C, H, W) tensor."""
    b = x.size(0)
    inv_s = 1.0 / float(scale)
    theta = torch.zeros((b, 2, 3), dtype=x.dtype, device=x.device)
    theta[:, 0, 0] = inv_s
    theta[:, 1, 1] = inv_s
    theta[:, 1, 2] = float(ty)
    grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
    return torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=False
    )


def build_tta_views(images: torch.Tensor, tta_views: int = 1) -> list[tuple[torch.Tensor, float]]:
    """Build weighted multi-view spatial/scale TTA views for inference evaluation."""
    if tta_views <= 1:
        return [(images, 1.0)]
    if tta_views == 2:
        return [(images, 0.55), (torch.flip(images, dims=[-1]), 0.45)]
    if tta_views == 5:
        return [
            (images, 0.36),
            (torch.flip(images, dims=[-1]), 0.16),
            (_shift_scale_batch(images, scale=1.0, ty=-0.16), 0.16),
            (_shift_scale_batch(images, scale=1.0, ty=0.16), 0.16),
            (_shift_scale_batch(images, scale=1.12, ty=0.0), 0.16),
        ]
    raise ValueError(f"Unsupported tta_views={tta_views}. Expected 1, 2, or 5.")


def decode_predictions_from_probs(probs: torch.Tensor, hierarchical_oa: bool = False) -> torch.Tensor:
    """Decode 5-class predictions with optional binary-OA probability coherence."""
    preds = probs.argmax(dim=1)
    if hierarchical_oa and probs.size(1) == 5:
        oa_prob = probs[:, 3:].sum(dim=1)
        promote_mask = (preds < 3) & (oa_prob > 0.50)
        if promote_mask.any():
            severe_pred = 3 + probs[promote_mask, 3:5].argmax(dim=1)
            preds = preds.clone()
            preds[promote_mask] = severe_pred
    return preds


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
    oa_threshold_kl: int = 3,
    criterion: nn.Module | None = None,
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
    tta_views: int = 1,
    hierarchical_oa: bool = False,
) -> dict[str, Any]:
    """Evaluate a trained classifier on validation or locked test data."""
    model.eval()
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    weighted_loss_sum = 0.0
    total_samples = 0
    class_loss_sum = [0.0 for _ in range(num_classes)]
    class_count = [0 for _ in range(num_classes)]

    for batch in loader:
        images, targets = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        views = build_tta_views(images, tta_views=tta_views)
        view0_logits = model(views[0][0])
        if len(views) == 1:
            probs = torch.softmax(view0_logits, dim=1)
        else:
            probs = torch.softmax(view0_logits, dim=1) * views[0][1]
            for view_x, view_w in views[1:]:
                probs = probs + torch.softmax(model(view_x), dim=1) * view_w

        if criterion is not None:
            dev_targets = targets.to(device, non_blocking=True)
            weighted_loss_sum += float(criterion(view0_logits, dev_targets).item()) * dev_targets.size(0)
            total_samples += int(dev_targets.size(0))
            per_sample = compute_per_sample_classification_loss(
                logits=view0_logits,
                targets=dev_targets,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
            )
            for c in range(num_classes):
                mask = dev_targets == c
                if mask.any():
                    class_loss_sum[c] += float(per_sample[mask].sum().item())
                    class_count[c] += int(mask.sum().item())
        preds = decode_predictions_from_probs(probs, hierarchical_oa=hierarchical_oa)
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=num_classes,
        oa_threshold_kl=oa_threshold_kl,
    )
    if criterion is not None:
        metrics["val_loss"] = float(weighted_loss_sum / max(1, total_samples))
        present_means = [
            class_loss_sum[c] / class_count[c]
            for c in range(num_classes)
            if class_count[c] > 0
        ]
        metrics["val_balanced_loss"] = float(sum(present_means) / max(1, len(present_means)))
    return metrics


def compute_selection_score(val_metrics: dict[str, Any], metric: str = "macro_f1") -> float:
    """Compute scalar validation selection objective (higher is always better)."""
    key = metric.lower()
    if key == "macro_f1":
        return float(val_metrics["macro_f1"])
    if key in ("oa_balanced", "oa_balanced_clean"):
        per_class = val_metrics["per_class_f1"]
        base = (
            0.40 * float(val_metrics["macro_f1"])
            + 0.25 * float(per_class["KL3"])
            + 0.25 * float(per_class["KL4"])
            + 0.10 * float(val_metrics["binary_oa_f1"])
        )
        if key == "oa_balanced_clean":
            base -= 0.10 * float(val_metrics.get("val_balanced_loss", 0.0))
        return base
    if key in ("val_loss", "val_balanced_loss"):
        return -float(val_metrics[key])
    raise ValueError(
        f"Unsupported selection_metric '{metric}'. "
        "Expected 'macro_f1', 'oa_balanced', 'oa_balanced_clean', 'val_loss', or 'val_balanced_loss'."
    )


def _find_final_linear_layer(model: nn.Module, num_classes: int) -> nn.Linear:
    """Locate the final KL classification linear projection for optional auxiliary training."""
    candidates = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == num_classes]
    if not candidates:
        raise ValueError(f"Could not find final nn.Linear(..., {num_classes}) layer in model.")
    return candidates[-1]


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
    class_weights: list[float] | torch.Tensor | None = None,
    num_classes: int = 5,
    oa_threshold_kl: int = 3,
    use_amp: bool = True,
    patience: int = 12,
    min_best_epoch: int = 1,
    selection_metric: str = "macro_f1",
    scheduler_type: str = "none",
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    min_lr: float = 1e-6,
    plateau_metric: str = "val_loss",
    optimizer_type: str = "adamw",
    mixup_alpha: float = 0.0,
    ordinal_weight: float = 0.0,
    ema_decay: float = 0.0,
    aux_subscore_weight: float = 0.0,
    oa_bce_weight: float = 0.0,
    oa_pos_weight: float = 1.6,
    severe_pair_weight: float = 0.0,
    severe_pos_weight: float = 0.85,
    tta_views: int = 1,
    hierarchical_oa: bool = False,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Train classifier on train_loader and select best weights via val_loader only."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    model = model.to(device)
    weight_tensor = (
        torch.as_tensor(class_weights, dtype=torch.float32)
        if class_weights is not None
        else None
    )
    criterion = build_classification_loss(
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        class_weights=weight_tensor,
    ).to(device)

    aux_head: nn.Linear | None = None
    aux_hook = None
    feat_buf: dict[str, torch.Tensor] = {}
    params: list[torch.nn.Parameter] = list(model.parameters())
    if aux_subscore_weight > 0.0:
        final_linear = _find_final_linear_layer(model, num_classes=num_classes)
        aux_head = nn.Linear(final_linear.in_features, 3).to(device)
        params.extend(aux_head.parameters())

        def _capture_features(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            feat_buf["x"] = inputs[0]

        aux_hook = final_linear.register_forward_pre_hook(_capture_features)

    opt_key = optimizer_type.lower()
    if opt_key not in ("adam", "adamw"):
        raise ValueError(f"Unsupported optimizer_type '{optimizer_type}'. Expected 'adam' or 'adamw'.")
    opt_cls = torch.optim.Adam if opt_key == "adam" else torch.optim.AdamW
    use_fused = bool(
        device.type == "cuda"
        and "fused" in inspect.signature(opt_cls).parameters
    )
    optimizer = opt_cls(
        params,
        lr=lr,
        weight_decay=weight_decay,
        fused=use_fused,
    )
    sched_key = scheduler_type.lower()
    if sched_key == "plateau":
        scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )
    elif sched_key == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=min_lr,
        )
    elif sched_key == "none":
        scheduler = None
    else:
        raise ValueError(
            f"Unsupported scheduler_type '{scheduler_type}'. Expected 'none', 'plateau', or 'cosine'."
        )

    amp_enabled = bool(use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported())
    best_score = float("-inf")
    best_epoch = 0
    no_improve = 0
    effective_min_best = max(1, min(int(min_best_epoch), int(epochs)))
    stop_warmup = max(10, effective_min_best)
    ema_state = (
        {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ema_decay > 0.0
        else None
    )
    best_state = deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []
    plateau_key = plateau_metric.lower()
    if plateau_key not in ("val_loss", "val_balanced_loss"):
        raise ValueError(
            f"Unsupported plateau_metric '{plateau_metric}'. Expected 'val_loss' or 'val_balanced_loss'."
        )
    grade_indices = torch.arange(num_classes, device=device, dtype=torch.float32)
    oa_pos_weight_tensor = torch.tensor(float(oa_pos_weight), dtype=torch.float32, device=device)
    severe_pos_weight_tensor = torch.tensor(float(severe_pos_weight), dtype=torch.float32, device=device)

    try:
        for epoch in range(1, epochs + 1):
            t_ep = time.perf_counter()
            model.train()
            if aux_head is not None:
                aux_head.train()
            running_loss = torch.zeros((), dtype=torch.float32, device=device)
            n_samples = 0
            for batch in train_loader:
                images, targets = batch[0], batch[1]
                aux_targets = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if mixup_alpha > 0.0 and images.size(0) > 1:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(images.size(0), device=device)
                    batch_inputs = lam * images + (1.0 - lam) * images[perm]
                    targets_b = targets[perm]
                else:
                    lam = 1.0
                    batch_inputs = images
                    targets_b = targets

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    logits = model(batch_inputs)
                    loss = (
                        lam * criterion(logits, targets) + (1.0 - lam) * criterion(logits, targets_b)
                        if lam < 1.0
                        else criterion(logits, targets)
                    )
                    if ordinal_weight > 0.0:
                        probs_f = torch.softmax(logits.float(), dim=1)
                        pred_grade = (probs_f * grade_indices).sum(dim=1)
                        target_grade = lam * targets.float() + (1.0 - lam) * targets_b.float()
                        loss = loss + ordinal_weight * torch.nn.functional.smooth_l1_loss(
                            pred_grade, target_grade
                        )
                    if oa_bce_weight > 0.0 and lam == 1.0:
                        logits_f = logits.float()
                        logit_non_oa = torch.logsumexp(logits_f[:, :oa_threshold_kl], dim=1)
                        logit_oa = torch.logsumexp(logits_f[:, oa_threshold_kl:], dim=1)
                        oa_target = (targets >= oa_threshold_kl).float()
                        loss = loss + float(oa_bce_weight) * torch.nn.functional.binary_cross_entropy_with_logits(
                            logit_oa - logit_non_oa,
                            oa_target,
                            pos_weight=oa_pos_weight_tensor,
                        )
                    if severe_pair_weight > 0.0 and lam == 1.0 and num_classes >= oa_threshold_kl + 2:
                        mask_34 = targets >= oa_threshold_kl
                        if mask_34.any():
                            logits_34 = logits.float()[mask_34]
                            pair_logit = logits_34[:, oa_threshold_kl + 1] - logits_34[:, oa_threshold_kl]
                            pair_target = (targets[mask_34] >= (oa_threshold_kl + 1)).float()
                            loss = loss + float(severe_pair_weight) * torch.nn.functional.binary_cross_entropy_with_logits(
                                pair_logit,
                                pair_target,
                                pos_weight=severe_pos_weight_tensor,
                            )
                    if aux_head is not None and aux_targets is not None and lam == 1.0:
                        mask = aux_targets[:, 3]
                        valid_sum = mask.sum()
                        if float(valid_sum.item()) > 0.0:
                            pred_aux = aux_head(feat_buf["x"].float())
                            loss_op = torch.nn.functional.smooth_l1_loss(
                                torch.sigmoid(pred_aux[:, 0]), aux_targets[:, 0], reduction="none"
                            )
                            loss_jsn = torch.nn.functional.smooth_l1_loss(
                                torch.sigmoid(pred_aux[:, 1]), aux_targets[:, 1], reduction="none"
                            )
                            loss_er = torch.nn.functional.binary_cross_entropy_with_logits(
                                pred_aux[:, 2], aux_targets[:, 2], reduction="none"
                            )
                            aux_loss = ((loss_op + loss_jsn + 0.5 * loss_er) * mask).sum() / valid_sum.clamp_min(1.0)
                            loss = loss + float(aux_subscore_weight) * aux_loss
                loss.backward()
                optimizer.step()
                if ema_state is not None:
                    cur_state = model.state_dict()
                    with torch.no_grad():
                        for k, ema_v in ema_state.items():
                            cur_v = cur_state[k]
                            if ema_v.is_floating_point():
                                ema_v.mul_(ema_decay).add_(cur_v.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_v.copy_(cur_v)
                running_loss += loss.detach().float() * targets.size(0)
                n_samples += targets.size(0)

            live_state = None
            if ema_state is not None:
                live_state = deepcopy(model.state_dict())
                model.load_state_dict(ema_state)

            val_metrics = evaluate_classifier(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=num_classes,
                oa_threshold_kl=oa_threshold_kl,
                criterion=criterion,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                tta_views=tta_views,
                hierarchical_oa=hierarchical_oa,
            )
            score = compute_selection_score(val_metrics, metric=selection_metric)
            val_metrics["selection_score"] = score
            if scheduler is not None:
                if sched_key == "plateau":
                    scheduler.step(float(val_metrics[plateau_key]))
                else:
                    scheduler.step()

            cur_lr = float(optimizer.param_groups[0]["lr"])
            train_loss = float(running_loss.item()) / max(1, n_samples)
            ep_sec = time.perf_counter() - t_ep
            history.append({"epoch": epoch, "lr": cur_lr, "train_loss": train_loss, "val": val_metrics})

            if epoch >= effective_min_best and score > best_score:
                best_score = score
                best_epoch = epoch
                no_improve = 0
                best_state = deepcopy(model.state_dict())
            elif epoch >= stop_warmup:
                no_improve += 1

            if live_state is not None:
                model.load_state_dict(live_state)

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:02d}/{epochs} ({ep_sec:.1f}s) | lr={cur_lr:.2e} | loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['val_loss']:.4f} | val_bal_loss={val_metrics['val_balanced_loss']:.4f} | "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
                    f"val_kl3_f1={val_metrics['per_class_f1']['KL3']:.4f} | "
                    f"val_kl4_f1={val_metrics['per_class_f1']['KL4']:.4f} | "
                    f"best_ep={best_epoch}"
                )

            if patience > 0 and epoch >= stop_warmup and no_improve >= patience:
                print(
                    f"Classifier early stopping at epoch {epoch} "
                    f"(best_ep={best_epoch}, {selection_metric}={best_score:.4f})"
                )
                break
    finally:
        if aux_hook is not None:
            aux_hook.remove()

    model.load_state_dict(best_state)
    return model, history
