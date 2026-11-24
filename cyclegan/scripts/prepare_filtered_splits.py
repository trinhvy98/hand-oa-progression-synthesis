"""CLI script to apply clinical Filter 1 and freeze patient-level train/val/test splits."""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hand_oa.data import (  # noqa: E402
    QualityFilterConfig,
    assign_patient_splits,
    build_manifest_from_excel,
    filter_manifest_by_quality,
    summarize_manifest,
    verify_no_data_leakage,
)
from hand_oa.utils import ensure_dir, load_yaml_config, resolve_project_path, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Filter 1 quality control before patient-level splitting."
    )
    parser.add_argument(
        "--config",
        default="configs/data_filtered.yaml",
        help="Path to filtered data YAML configuration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run Filter 1 and overwrite manifest_filtered.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml_config(resolve_project_path(args.config, _ROOT))
    paths_cfg = data_cfg["paths"]
    dataset_cfg = data_cfg["dataset"]
    split_cfg = data_cfg["split"]
    qf_raw = dict(data_cfg.get("quality_filter", {}))

    manifest_path = resolve_project_path(paths_cfg["manifest_path"], _ROOT)
    summary_path = manifest_path.with_suffix(".summary.json")
    if manifest_path.stem == "manifest_filtered":
        report_path = manifest_path.with_name("filter1_report.json")
        audit_path = manifest_path.with_name("filter1_audit.csv")
    else:
        report_path = manifest_path.with_name(f"{manifest_path.stem}_filter1_report.json")
        audit_path = manifest_path.with_name(f"{manifest_path.stem}_filter1_audit.csv")

    if manifest_path.is_file() and not args.overwrite:
        import pandas as pd

        existing = pd.read_csv(manifest_path, dtype={"patient_id": str, "sample_id": str, "filename": str})
        verify_no_data_leakage(existing)
        print(json.dumps(summarize_manifest(existing), indent=2))
        return

    images_dir = resolve_project_path(paths_cfg["images_dir"], _ROOT)
    excel_path = resolve_project_path(paths_cfg["excel_path"], _ROOT)

    raw_df = build_manifest_from_excel(
        excel_path=excel_path,
        images_dir=images_dir,
        joints=dataset_cfg["joints"],
        visit=dataset_cfg["visit"],
        valid_kl_grades=dataset_cfg["valid_kl_grades"],
    )

    qf_cfg = QualityFilterConfig(**qf_raw)
    kept_df, audit_df, filter_report = filter_manifest_by_quality(
        manifest_df=raw_df,
        images_dir=images_dir,
        cfg=qf_cfg,
    )

    split_df = assign_patient_splits(
        kept_df,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["seed"]),
    )
    summary = summarize_manifest(split_df)

    ensure_dir(manifest_path.parent)
    split_df.to_csv(manifest_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    save_json(summary, summary_path)
    save_json({**filter_report, "split_summary": summary}, report_path)

    print("Filter 1 report:")
    print(json.dumps(filter_report, indent=2))
    print("\nFiltered split summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
