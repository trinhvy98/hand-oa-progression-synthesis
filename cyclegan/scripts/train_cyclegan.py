"""CLI entry point for single-GPU or multi-GPU (torchrun DDP) CycleGAN training."""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cyclegan.engine import CycleGANTrainer  # noqa: E402
from hand_oa.utils import (  # noqa: E402
    cleanup_distributed,
    init_distributed,
    is_main_process,
    load_yaml_config,
    merge_configs,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CycleGAN for KL0 -> KL3 synthesis.")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to data YAML.")
    parser.add_argument("--config", default="configs/cyclegan.yaml", help="Path to CycleGAN YAML.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override global batch size.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps/epoch.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--amp", dest="use_amp", action="store_true", default=None, help="Enable AMP.")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Disable AMP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_distributed, rank, local_rank, world_size, device = init_distributed()
    try:
        data_cfg = load_yaml_config(resolve_project_path(args.data_config, _ROOT))
        cg_cfg = load_yaml_config(resolve_project_path(args.config, _ROOT))

        overrides: dict[str, object] = {}
        if args.batch_size is not None:
            overrides.setdefault("data", {})["batch_size"] = args.batch_size
        if args.output_dir is not None:
            overrides.setdefault("experiment", {})["output_dir"] = args.output_dir
        if args.use_amp is not None:
            overrides.setdefault("training", {})["use_amp"] = args.use_amp
        if overrides:
            cg_cfg = merge_configs(cg_cfg, overrides)

        trainer = CycleGANTrainer(
            data_cfg=data_cfg,
            cyclegan_cfg=cg_cfg,
            project_root=_ROOT,
            device=device,
            is_distributed=is_distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )
        if is_main_process(rank):
            print(
                f"Training initialized: world_size={world_size}, device={device}, "
                f"global_batch={cg_cfg['data']['batch_size']}, amp={trainer.model.use_amp} "
                f"({trainer.model.amp_dtype})"
            )
        history = trainer.fit(
            max_epochs=args.epochs,
            max_steps_per_epoch=args.max_steps,
            resume_checkpoint=args.resume,
        )
        if is_main_process(rank):
            print(f"Training completed ({len(history)} epochs). Outputs saved to: {trainer.output_dir}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
