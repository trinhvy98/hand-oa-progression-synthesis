"""CLI script to generate and verify the frozen patient-level train/val/test split."""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hand_oa.data import load_manifest_from_config, summarize_manifest  # noqa: E402
from hand_oa.utils import load_yaml_config, resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leak-free patient-level splits.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data YAML config.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate manifest.csv even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml_config(resolve_project_path(args.config, _ROOT))
    manifest, _ = load_manifest_from_config(data_cfg, _ROOT, overwrite=args.overwrite)
    summary = summarize_manifest(manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
