# Diffusion Track (Phase 2)

This directory is pre-configured for implementing Diffusion-based Hand OA progression synthesis (e.g., Conditional DDPM / Img2Img / SDEdit using `diffusers` and PyTorch) after completing the CycleGAN baseline.

## Shared Components Reused from `src/hand_oa/`

To ensure a fair, leak-free comparison against CycleGAN, Diffusion should **not** duplicate data splitting or evaluation code. Reuse directly:

- `hand_oa.data.load_manifest_from_config`: Loads the exact same frozen patient-level `train` / `val` / `test` split (`data/processed/manifest.csv`).
- `hand_oa.data.UnpairedDomainDataset` / `SingleDomainDataset`: Automatically blocks `split="test"` from entering training or synthetic data generation.
- `hand_oa.data.build_generative_transform` & `denormalize_to_uint8`: Identical `[-1, 1]` grayscale preprocessing.
- `evaluation/run_downstream_eval.py`: Accepts `--synthetic-manifest` from Diffusion (same schema as CycleGAN: `image_path`, `target_kl`, `source_split`) to evaluate on the identical locked `test` set.
