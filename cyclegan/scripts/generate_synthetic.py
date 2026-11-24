"""CLI entry point for generating synthetic OA images from a trained CycleGAN checkpoint."""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cyclegan.engine import generate_synthetic_images  # noqa: E402
from hand_oa.utils import load_yaml_config, merge_configs, resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize Hand OA images using CycleGAN.")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to data YAML.")
    parser.add_argument("--config", default="configs/cyclegan.yaml", help="Path to CycleGAN YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained CycleGAN checkpoint.")
    parser.add_argument(
        "--source-split",
        default=None,
        choices=["train", "val"],
        help="Source pool split (only 'train' or 'val'; 'test' is strictly forbidden).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum images to generate.")
    parser.add_argument("--output-dir", default=None, help="Optional output folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml_config(resolve_project_path(args.data_config, _ROOT))
    cg_cfg = load_yaml_config(resolve_project_path(args.config, _ROOT))

    overrides: dict[str, object] = {}
    if args.source_split is not None:
        overrides.setdefault("generation", {})["source_split"] = args.source_split
    if args.max_samples is not None:
        overrides.setdefault("generation", {})["max_samples"] = args.max_samples
    if overrides:
        cg_cfg = merge_configs(cg_cfg, overrides)

    manifest_out = generate_synthetic_images(
        data_cfg=data_cfg,
        cyclegan_cfg=cg_cfg,
        checkpoint_path=args.checkpoint,
        project_root=_ROOT,
        output_dir=args.output_dir,
    )
    print(f"Synthetic manifest written to: {manifest_out}")


if __name__ == "__main__":
    main()
