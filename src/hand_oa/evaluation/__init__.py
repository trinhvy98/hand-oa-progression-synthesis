"""Shared downstream classification evaluation utilities."""

from hand_oa.evaluation.classifier import (
    build_classifier,
    compute_selection_score,
    evaluate_classifier,
    train_classifier,
)
from hand_oa.evaluation.fast_gpu_loader import (
    FastPreloadedBatchLoader,
    build_fast_classification_loaders,
)
from hand_oa.evaluation.generative_metrics import (
    InceptionFeatureExtractor,
    compute_batch_ssim,
    compute_fid_from_features,
    compute_kid_from_features,
    evaluate_translation_validation,
    extract_features_from_loader,
)
from hand_oa.evaluation.losses import (
    FocalLoss,
    build_classification_loss,
    compute_per_sample_classification_loss,
)
from hand_oa.evaluation.metrics import compute_classification_metrics

__all__ = [
    "FastPreloadedBatchLoader",
    "FocalLoss",
    "InceptionFeatureExtractor",
    "build_classification_loss",
    "compute_per_sample_classification_loss",
    "build_classifier",
    "build_fast_classification_loaders",
    "compute_batch_ssim",
    "compute_classification_metrics",
    "compute_fid_from_features",
    "compute_kid_from_features",
    "compute_selection_score",
    "evaluate_classifier",
    "evaluate_translation_validation",
    "extract_features_from_loader",
    "train_classifier",
]
