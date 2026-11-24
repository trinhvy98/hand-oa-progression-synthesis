"""CycleGAN composite model supporting single-GPU, DDP multi-GPU, and AMP training."""

import itertools
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from cyclegan.models.networks import (
    GANLoss,
    ImagePool,
    PatchGANDiscriminator,
    ResNetGenerator,
    init_weights,
)


def _set_requires_grad(nets: list[nn.Module], requires_grad: bool) -> None:
    """Toggle parameter gradients to skip unused backward graphs."""
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def _unwrap_model(net: nn.Module) -> nn.Module:
    """Return underlying module if wrapped in DistributedDataParallel."""
    return net.module if isinstance(net, DDP) else net


def _build_linear_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    decay_start_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Keep LR constant until `decay_start_epoch`, then linearly decay toward zero."""
    decay_span = max(1, total_epochs - decay_start_epoch)

    def _lr_lambda(epoch: int) -> float:
        if epoch < decay_start_epoch:
            return 1.0
        remaining = max(0.0, 1.0 - float(epoch - decay_start_epoch + 1) / float(decay_span))
        return remaining

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


class CycleGANModel:
    """Complete unpaired CycleGAN model for A<->B radiographic translation."""

    def __init__(
        self,
        in_channels: int = 1,
        ngf: int = 64,
        ndf: int = 64,
        n_res_blocks: int = 9,
        gan_mode: str = "lsgan",
        pool_size: int = 50,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_cycle_a: float = 10.0,
        lambda_cycle_b: float = 10.0,
        lambda_identity: float = 0.0,
        epochs: int = 200,
        decay_start_epoch: int = 100,
        use_amp: bool = False,
        is_distributed: bool = False,
        local_rank: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = is_distributed
        self.lambda_cycle_a = lambda_cycle_a
        self.lambda_cycle_b = lambda_cycle_b
        self.lambda_identity = lambda_identity

        self.use_amp = bool(use_amp and self.device.type == "cuda")
        if self.use_amp and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.scaler: torch.amp.GradScaler | None = None
        elif self.use_amp:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        g_a2b = ResNetGenerator(in_channels, in_channels, ngf, n_res_blocks).to(self.device)
        g_b2a = ResNetGenerator(in_channels, in_channels, ngf, n_res_blocks).to(self.device)
        d_a = PatchGANDiscriminator(in_channels, ndf).to(self.device)
        d_b = PatchGANDiscriminator(in_channels, ndf).to(self.device)

        for net in (g_a2b, g_b2a, d_a, d_b):
            init_weights(net)

        if self.is_distributed:
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if self.device.type == "cuda" else {}
            self.net_g_a2b: nn.Module = DDP(g_a2b, **ddp_kwargs)
            self.net_g_b2a: nn.Module = DDP(g_b2a, **ddp_kwargs)
            self.net_d_a: nn.Module = DDP(d_a, **ddp_kwargs)
            self.net_d_b: nn.Module = DDP(d_b, **ddp_kwargs)
        else:
            self.net_g_a2b = g_a2b
            self.net_g_b2a = g_b2a
            self.net_d_a = d_a
            self.net_d_b = d_b

        self.fake_a_pool = ImagePool(pool_size=pool_size)
        self.fake_b_pool = ImagePool(pool_size=pool_size)

        self.criterion_gan = GANLoss(gan_mode=gan_mode).to(self.device)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.optimizer_g = torch.optim.Adam(
            itertools.chain(self.net_g_a2b.parameters(), self.net_g_b2a.parameters()),
            lr=lr,
            betas=(beta1, beta2),
        )
        self.optimizer_d = torch.optim.Adam(
            itertools.chain(self.net_d_a.parameters(), self.net_d_b.parameters()),
            lr=lr,
            betas=(beta1, beta2),
        )

        self.scheduler_g = _build_linear_decay_scheduler(
            self.optimizer_g, total_epochs=epochs, decay_start_epoch=decay_start_epoch
        )
        self.scheduler_d = _build_linear_decay_scheduler(
            self.optimizer_d, total_epochs=epochs, decay_start_epoch=decay_start_epoch
        )

    def train_mode(self) -> None:
        for net in (self.net_g_a2b, self.net_g_b2a, self.net_d_a, self.net_d_b):
            net.train()

    def eval_mode(self) -> None:
        for net in (self.net_g_a2b, self.net_g_b2a, self.net_d_a, self.net_d_b):
            net.eval()

    def _compute_discriminator_loss(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake_from_pool: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate along batch dim so DDP runs exactly one forward per backward pass;
        # InstanceNorm2d normalizes per-sample independently so outputs match separate calls.
        combined = torch.cat([real, fake_from_pool.detach()], dim=0)
        preds = discriminator(combined)
        pred_real, pred_fake = preds.chunk(2, dim=0)
        loss_real = self.criterion_gan(pred_real, target_is_real=True)
        loss_fake = self.criterion_gan(pred_fake, target_is_real=False)
        # Scale by 0.5 per Zhu et al. (2017) Section 7.1
        return 0.5 * (loss_real + loss_fake)

    def _backward_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
        else:
            loss.backward()
            optimizer.step()

    def train_step(self, real_a: torch.Tensor, real_b: torch.Tensor) -> dict[str, float]:
        """Execute one CycleGAN optimization step for generators and discriminators."""
        self.train_mode()
        real_a = real_a.to(self.device, non_blocking=True)
        real_b = real_b.to(self.device, non_blocking=True)

        # 1. Update Generators (use unwrapped D during G step so DDP doesn't expect D backward yet)
        d_a_eval = _unwrap_model(self.net_d_a)
        d_b_eval = _unwrap_model(self.net_d_b)
        _set_requires_grad([self.net_d_a, self.net_d_b], False)
        self.optimizer_g.zero_grad()

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            fake_b = self.net_g_a2b(real_a)
            rec_a = self.net_g_b2a(fake_b)
            fake_a = self.net_g_b2a(real_b)
            rec_b = self.net_g_a2b(fake_a)

            loss_gan_a2b = self.criterion_gan(d_b_eval(fake_b), target_is_real=True)
            loss_gan_b2a = self.criterion_gan(d_a_eval(fake_a), target_is_real=True)
            loss_cycle_a = self.criterion_cycle(rec_a, real_a) * self.lambda_cycle_a
            loss_cycle_b = self.criterion_cycle(rec_b, real_b) * self.lambda_cycle_b

            loss_idt_a = torch.tensor(0.0, device=self.device)
            loss_idt_b = torch.tensor(0.0, device=self.device)
            if self.lambda_identity > 0.0:
                idt_a = self.net_g_b2a(real_a)
                idt_b = self.net_g_a2b(real_b)
                loss_idt_a = self.criterion_identity(idt_a, real_a) * self.lambda_cycle_a * self.lambda_identity
                loss_idt_b = self.criterion_identity(idt_b, real_b) * self.lambda_cycle_b * self.lambda_identity

            loss_g = loss_gan_a2b + loss_gan_b2a + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b

        self._backward_and_step(loss_g, self.optimizer_g)

        # 2. Update Discriminators (D_A and D_B)
        _set_requires_grad([self.net_d_a, self.net_d_b], True)
        self.optimizer_d.zero_grad()

        pooled_a = self.fake_a_pool.query(fake_a.detach())
        pooled_b = self.fake_b_pool.query(fake_b.detach())
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            loss_d_a = self._compute_discriminator_loss(self.net_d_a, real_a, pooled_a)
            loss_d_b = self._compute_discriminator_loss(self.net_d_b, real_b, pooled_b)
            loss_d = loss_d_a + loss_d_b

        self._backward_and_step(loss_d, self.optimizer_d)
        if self.scaler is not None:
            self.scaler.update()

        return {
            "loss_g": float(loss_g.item()),
            "loss_gan_a2b": float(loss_gan_a2b.item()),
            "loss_gan_b2a": float(loss_gan_b2a.item()),
            "loss_cycle_a": float(loss_cycle_a.item()),
            "loss_cycle_b": float(loss_cycle_b.item()),
            "loss_idt": float((loss_idt_a + loss_idt_b).item()),
            "loss_d_a": float(loss_d_a.item()),
            "loss_d_b": float(loss_d_b.item()),
        }

    @torch.no_grad()
    def translate(self, images: torch.Tensor, direction: str = "a2b") -> torch.Tensor:
        """Translate a batch of images using the specified generator direction."""
        self.eval_mode()
        images = images.to(self.device, non_blocking=True)
        g_a2b = _unwrap_model(self.net_g_a2b)
        g_b2a = _unwrap_model(self.net_g_b2a)
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            if direction == "a2b":
                return g_a2b(images).float()
            if direction == "b2a":
                return g_b2a(images).float()
        raise ValueError(f"Invalid translation direction '{direction}'. Expected 'a2b' or 'b2a'.")

    @torch.no_grad()
    def preview_cycles(
        self, real_a: torch.Tensor, real_b: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Generate forward and reconstructed images for visual checkpoint inspection."""
        self.eval_mode()
        real_a = real_a.to(self.device)
        real_b = real_b.to(self.device)
        g_a2b = _unwrap_model(self.net_g_a2b)
        g_b2a = _unwrap_model(self.net_g_b2a)
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            fake_b = g_a2b(real_a)
            rec_a = g_b2a(fake_b)
            fake_a = g_b2a(real_b)
            rec_b = g_a2b(fake_a)
        return {
            "real_a": real_a.float().cpu(),
            "fake_b": fake_b.float().cpu(),
            "rec_a": rec_a.float().cpu(),
            "real_b": real_b.float().cpu(),
            "fake_a": fake_a.float().cpu(),
            "rec_b": rec_b.float().cpu(),
        }

    def step_schedulers(self) -> float:
        """Advance LR schedulers once per epoch and return current generator LR."""
        self.scheduler_g.step()
        self.scheduler_d.step()
        return float(self.optimizer_g.param_groups[0]["lr"])

    def save_checkpoint(self, path: str | Path, epoch: int, extra: dict[str, Any] | None = None) -> Path:
        """Persist unwrapped weights and optimizer state for portable resume/inference."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "net_g_a2b": _unwrap_model(self.net_g_a2b).state_dict(),
            "net_g_b2a": _unwrap_model(self.net_g_b2a).state_dict(),
            "net_d_a": _unwrap_model(self.net_d_a).state_dict(),
            "net_d_b": _unwrap_model(self.net_d_b).state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, target)
        return target

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = False) -> int:
        """Restore weights (and optionally optimizers) from checkpoint, returning saved epoch."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        _unwrap_model(self.net_g_a2b).load_state_dict(payload["net_g_a2b"])
        _unwrap_model(self.net_g_b2a).load_state_dict(payload["net_g_b2a"])
        _unwrap_model(self.net_d_a).load_state_dict(payload["net_d_a"])
        _unwrap_model(self.net_d_b).load_state_dict(payload["net_d_b"])
        if load_optimizers and "optimizer_g" in payload:
            self.optimizer_g.load_state_dict(payload["optimizer_g"])
            self.optimizer_d.load_state_dict(payload["optimizer_d"])
            self.scheduler_g.load_state_dict(payload["scheduler_g"])
            self.scheduler_d.load_state_dict(payload["scheduler_d"])
        return int(payload.get("epoch", 0))
