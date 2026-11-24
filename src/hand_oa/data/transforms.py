"""Shared image transformations for generative models and downstream classifiers."""

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_generative_transform(
    image_size: int = 256,
    in_channels: int = 1,
    is_train: bool = True,
) -> transforms.Compose:
    """Build preprocessing pipeline normalizing X-rays into [-1, 1] for GAN/Diffusion models."""
    if in_channels not in (1, 3):
        raise ValueError(f"in_channels must be 1 or 3, got {in_channels}")

    pil_mode = "L" if in_channels == 1 else "RGB"
    mean = [0.5] * in_channels
    std = [0.5] * in_channels

    ops: list[object] = [
        transforms.Lambda(lambda img: img.convert(pil_mode)),
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    ]
    if is_train:
        # Left and right hands mirror anatomically, so horizontal flip is label-preserving
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def build_classifier_transform(
    image_size: int = 224,
    is_train: bool = True,
    skip_resize: bool = False,
) -> transforms.Compose:
    """Build ImageNet-normalized 3-channel transform for downstream VGG19/ResNet classifiers."""
    ops: list[object] = []
    if not skip_resize:
        ops.extend(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            ]
        )
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def denormalize_to_uint8(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized [-1, 1] single-image tensor (C, H, W) to a PIL Image."""
    img = tensor.detach().cpu().clamp(-1.0, 1.0)
    img = ((img + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
    if img.ndim == 3 and img.shape[0] == 1:
        return Image.fromarray(img.squeeze(0).numpy(), mode="L")
    if img.ndim == 3 and img.shape[0] == 3:
        return Image.fromarray(img.permute(1, 2, 0).numpy(), mode="RGB")
    raise ValueError(f"Expected image tensor shape (1|3, H, W), got {tuple(tensor.shape)}")
