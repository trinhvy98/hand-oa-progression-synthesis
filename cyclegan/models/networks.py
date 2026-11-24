"""CycleGAN generator, PatchGAN discriminator, GAN loss, and image history buffer."""

import random

import torch
import torch.nn as nn


def init_weights(net: nn.Module, init_gain: float = 0.02) -> None:
    """Initialize convolutional and normalization layers with N(0, 0.02) per Zhu et al. (2017)."""

    def _init_fn(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if hasattr(module, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            nn.init.normal_(module.weight.data, 0.0, init_gain)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1 and module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, init_gain)
            nn.init.constant_(module.bias.data, 0.0)

    net.apply(_init_fn)


class ResidualBlock(nn.Module):
    """Standard CycleGAN residual block with reflection padding and instance normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet-based CycleGAN generator (6 blocks for 128px, 9 blocks for >=256px)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        ngf: int = 64,
        n_res_blocks: int = 9,
    ) -> None:
        super().__init__()
        if n_res_blocks < 0:
            raise ValueError(f"n_res_blocks must be >= 0, got {n_res_blocks}")

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Two downsampling stages: ngf -> 2*ngf -> 4*ngf
        curr_dim = ngf
        for _ in range(2):
            next_dim = curr_dim * 2
            layers.extend(
                [
                    nn.Conv2d(curr_dim, next_dim, kernel_size=3, stride=2, padding=1, bias=True),
                    nn.InstanceNorm2d(next_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            curr_dim = next_dim

        # Residual bottleneck blocks
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock(curr_dim))

        # Two upsampling stages: 4*ngf -> 2*ngf -> ngf
        for _ in range(2):
            next_dim = curr_dim // 2
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        curr_dim,
                        next_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(next_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            curr_dim = next_dim

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (Zhu et al., ICCV 2017)."""

    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers.extend(
                [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers.extend(
            [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ImagePool:
    """History buffer storing previously generated images to stabilize discriminator updates."""

    def __init__(self, pool_size: int = 50) -> None:
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size <= 0:
            return images
        selected: list[torch.Tensor] = []
        for image in images:
            single = torch.unsqueeze(image.data, 0)
            if len(self.images) < self.pool_size:
                self.images.append(single)
                selected.append(single)
            elif random.random() > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                cached = self.images[idx].clone()
                self.images[idx] = single
                selected.append(cached)
            else:
                selected.append(single)
        return torch.cat(selected, dim=0)


class GANLoss(nn.Module):
    """Adversarial objective supporting LSGAN (MSE) and vanilla (BCEWithLogits)."""

    def __init__(self, gan_mode: str = "lsgan") -> None:
        super().__init__()
        mode = gan_mode.lower()
        if mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported gan_mode '{gan_mode}'. Expected 'lsgan' or 'vanilla'.")

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_val = 1.0 if target_is_real else 0.0
        target = torch.full_like(prediction, fill_value=target_val)
        return self.loss(prediction, target)
