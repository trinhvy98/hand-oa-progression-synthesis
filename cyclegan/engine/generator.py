"""Synthetic OA progression generation strictly from train/val source pools."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cyclegan.engine.trainer import build_cyclegan_from_config
from hand_oa.data import (
    SingleDomainDataset,
    build_generative_transform,
    denormalize_to_uint8,
    load_manifest_from_config,
)
from hand_oa.utils import ensure_dir, resolve_project_path


def generate_synthetic_images(
    data_cfg: dict[str, Any],
    cyclegan_cfg: dict[str, Any],
    checkpoint_path: str | Path,
    project_root: str | Path,
    output_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> Path:
    """Synthesize progression images from a trained CycleGAN checkpoint without touching test data."""
    root = Path(project_root)
    gen_cfg = cyclegan_cfg["generation"]
    cg_data = cyclegan_cfg["data"]

    source_split = str(gen_cfg["source_split"])
    if source_split not in ("train", "val"):
        raise ValueError(
            f"generation.source_split must be 'train' or 'val' to prevent test leakage, got '{source_split}'"
        )

    direction = str(gen_cfg["direction"])
    source_grades = cg_data["domain_a_grades"] if direction == "a2b" else cg_data["domain_b_grades"]
    target_label = int(gen_cfg["target_label"])

    dataset_cfg = data_cfg["dataset"]
    manifest, images_dir = load_manifest_from_config(data_cfg, root)

    transform = build_generative_transform(
        image_size=int(cg_data["image_size"]),
        in_channels=int(cg_data["in_channels"]),
        is_train=False,
    )
    source_dataset = SingleDomainDataset(
        manifest_df=manifest,
        images_dir=images_dir,
        split=source_split,
        grades=source_grades,
        joints=cg_data.get("joints", dataset_cfg["joints"]),
        transform=transform,
        max_samples=gen_cfg.get("max_samples"),
        seed=int(cyclegan_cfg["experiment"].get("seed", 42)),
    )
    loader = DataLoader(source_dataset, batch_size=int(cg_data["batch_size"]), shuffle=False)

    model = build_cyclegan_from_config(cyclegan_cfg, device=device)
    ckpt_epoch = model.load_checkpoint(resolve_project_path(checkpoint_path, root), load_optimizers=False)

    base_out = (
        Path(output_dir)
        if output_dir is not None
        else resolve_project_path(cyclegan_cfg["experiment"]["output_dir"], root)
        / str(gen_cfg["output_subdir"])
    )
    syn_img_dir = ensure_dir(base_out / "images")

    records: list[dict[str, object]] = []
    for batch in loader:
        translated = model.translate(batch["image"], direction=direction)
        for idx, sample_id in enumerate(batch["sample_id"]):
            out_name = f"{sample_id}_syn_kl{target_label}.png"
            out_path = syn_img_dir / out_name
            denormalize_to_uint8(translated[idx]).save(out_path)
            records.append(
                {
                    "sample_id": f"{sample_id}_syn_kl{target_label}",
                    "source_sample_id": sample_id,
                    "source_patient_id": batch["patient_id"][idx],
                    "source_split": source_split,
                    "source_kl": int(batch["kl_grade"][idx]),
                    "target_kl": target_label,
                    "checkpoint_epoch": ckpt_epoch,
                    "image_path": str(out_path.resolve()),
                }
            )

    manifest_out = base_out / "synthetic_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_out, index=False)
    return manifest_out
