"""Evaluate and rank saved CycleGAN checkpoints strictly on the validation split."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cyclegan.engine.trainer import build_cyclegan_from_config  # noqa: E402
from hand_oa.data import SingleDomainDataset, build_generative_transform, load_manifest_from_config  # noqa: E402
from hand_oa.evaluation import (  # noqa: E402
    InceptionFeatureExtractor,
    evaluate_translation_validation,
    extract_features_from_loader,
)
from hand_oa.utils import load_yaml_config, resolve_project_path, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CycleGAN checkpoints on val split.")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to data YAML.")
    parser.add_argument("--config", default="configs/cyclegan.yaml", help="Path to CycleGAN YAML.")
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        help="Directory with epoch_*.pt checkpoints (defaults to output_dir/checkpoints).",
    )
    parser.add_argument(
        "--copy-best",
        action="store_true",
        help="Copy winning checkpoint to checkpoints/best_checkpoint.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml_config(resolve_project_path(args.data_config, _ROOT))
    cg_cfg = load_yaml_config(resolve_project_path(args.config, _ROOT))

    manifest, images_dir = load_manifest_from_config(data_cfg, _ROOT)
    cg_data = cg_cfg["data"]
    es_cfg = cg_cfg.get("early_stopping", {})
    joints = cg_data.get("joints", data_cfg["dataset"]["joints"])
    val_tf = build_generative_transform(
        image_size=int(cg_data["image_size"]),
        in_channels=int(cg_data["in_channels"]),
        is_train=False,
    )

    val_src = SingleDomainDataset(
        manifest_df=manifest,
        images_dir=images_dir,
        split="val",
        grades=cg_data["domain_a_grades"],
        joints=joints,
        transform=val_tf,
    )
    val_tgt = SingleDomainDataset(
        manifest_df=manifest,
        images_dir=images_dir,
        split="val",
        grades=cg_data["domain_b_grades"],
        joints=joints,
        transform=val_tf,
    )
    src_loader = DataLoader(val_src, batch_size=32, shuffle=False, num_workers=0)
    tgt_loader = DataLoader(val_tgt, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = InceptionFeatureExtractor(
        device=device,
        pretrained=bool(es_cfg.get("use_pretrained_inception", True)),
    )
    tgt_feats = extract_features_from_loader(extractor, tgt_loader, image_key="image")

    out_dir = resolve_project_path(cg_cfg["experiment"]["output_dir"], _ROOT)
    ckpt_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else out_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No epoch_*.pt checkpoints found in {ckpt_dir}")

    min_cycle_ssim = float(es_cfg.get("min_cycle_ssim", 0.75))
    min_delta_l1 = float(es_cfg.get("min_delta_l1", 0.015))

    model = build_cyclegan_from_config(cg_cfg, device=device)
    rows: list[dict[str, object]] = []
    for ckpt in ckpts:
        epoch = model.load_checkpoint(ckpt, load_optimizers=False)
        metrics = evaluate_translation_validation(
            forward_fn=lambda x: model.translate(x, direction="a2b"),
            backward_fn=lambda x: model.translate(x, direction="b2a"),
            val_source_loader=src_loader,
            real_target_feats=tgt_feats,
            extractor=extractor,
            device=device,
        )
        qualified = (
            metrics["val_cycle_ssim"] >= min_cycle_ssim
            and metrics["val_delta_l1"] >= min_delta_l1
        )
        rows.append({"epoch": epoch, "checkpoint": str(ckpt), "qualified": qualified, **metrics})

    df = pd.DataFrame(rows).sort_values(["qualified", "val_kid"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(out_dir / "val_checkpoint_ranking.csv", index=False)

    best_row = df.iloc[0].to_dict()
    if args.copy_best:
        shutil.copy2(best_row["checkpoint"], ckpt_dir / "best_checkpoint.pt")
        save_json(best_row, out_dir / "best_checkpoint_info.json")
    print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
