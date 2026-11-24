"""Shared data loading, patient-level splitting, transforms, and PyTorch datasets."""

from hand_oa.data.datasets import (
    JointClassificationDataset,
    SingleDomainDataset,
    UnpairedDomainDataset,
)
from hand_oa.data.metadata import build_manifest_from_excel, load_excel_table
from hand_oa.data.quality_filter import (
    QualityFilterConfig,
    compute_laplacian_variance,
    filter_manifest_by_quality,
    inspect_image_quality,
)
from hand_oa.data.split import (
    assign_patient_splits,
    load_manifest_from_config,
    load_or_create_manifest,
    summarize_manifest,
    verify_no_data_leakage,
)
from hand_oa.data.transforms import (
    build_classifier_transform,
    build_generative_transform,
    denormalize_to_uint8,
)

__all__ = [
    "JointClassificationDataset",
    "QualityFilterConfig",
    "SingleDomainDataset",
    "UnpairedDomainDataset",
    "assign_patient_splits",
    "build_classifier_transform",
    "build_generative_transform",
    "build_manifest_from_excel",
    "compute_laplacian_variance",
    "denormalize_to_uint8",
    "filter_manifest_by_quality",
    "inspect_image_quality",
    "load_excel_table",
    "load_manifest_from_config",
    "load_or_create_manifest",
    "summarize_manifest",
    "verify_no_data_leakage",
]
