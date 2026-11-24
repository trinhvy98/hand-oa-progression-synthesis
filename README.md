# Hand Osteoarthritis Progression Synthesis

Modular PyTorch research repository for synthesizing Hand Osteoarthritis (HOA) radiographic progression across Kellgren-Lawrence (KL) severity grades using **CycleGAN** and (next) **Diffusion** models, evaluated fairly on downstream KL-grade classification without data leakage.

## Repository Layout

```text
hand-oa-progression-synthesis/
├── configs/                 # Shared & model-specific YAML configurations
├── src/hand_oa/             # Shared package: data loading, patient-level splits, evaluation
├── cyclegan/                # CycleGAN track (models, trainer, generator, CLI scripts, SCC notebook)
├── diffusion/               # Next-phase Diffusion track (reuses src/hand_oa/ directly)
├── evaluation/              # Shared downstream VGG19/ResNet evaluation on locked test set
├── jobs/                    # BU SCC batch job (.qsub) templates
├── tests/                   # Automated unit tests (leakage assertions & smoke tests)
├── data/                    # (gitignored) hand.xlsx, Finger Joints/, data/processed/manifest.csv
├── docs/                    # (gitignored) Coursework documentation
└── references/              # (gitignored) Reference PDFs (Zhu et al. 2017, Tank et al. 2026)
```

## Anti-Data-Leakage Protocol

1. **Deduplication & Patient-Level Grouping**: `hand.xlsx` is deduplicated by patient `id`, and all joints belonging to the same patient (`dip2..5`, `pip2..5`, `mcp2..5`) are strictly assigned to a single split (`train`, `val`, or `test`).
2. **Three-Way Frozen Split**:
   - `test` (15% of patients): Locked never-seen holdout set. Neither CycleGAN nor Diffusion nor classifier tuning may touch `test` images.
   - `val` (8.5% of patients, i.e., 10% of the remaining 85%): Validation & internal generative source pool.
   - `train` (76.5% of patients, i.e., 90% of the remaining 85%): Training set for generative models and classifiers.
3. **Runtime Guards**: Both `UnpairedDomainDataset` and the synthetic generator explicitly reject `split="test"` for training or synthetic data augmentation.

## BU SCC Environment Setup

```bash
module load miniconda academic-ml/fall-2026
conda activate fall-2026-pyt
```

## Quick Start (CycleGAN)

1. **Prepare the frozen patient-level split manifest (run once, shared by CycleGAN & Diffusion):**
   ```bash
   python cyclegan/scripts/prepare_splits.py --config configs/data.yaml
   ```
2. **Train CycleGAN (e.g., KL0 -> KL3 on DIP joints):**
   ```bash
   python cyclegan/scripts/train_cyclegan.py --data-config configs/data.yaml --config configs/cyclegan.yaml
   ```
3. **Generate synthetic KL3 images (strictly from `train`/`val` source pools, never `test`):**
   ```bash
   python cyclegan/scripts/generate_synthetic.py --config configs/cyclegan.yaml --checkpoint outputs/cyclegan/checkpoints/epoch_0100.pt
   ```
4. **Evaluate downstream classifier fairly on the locked `test` set:**
   ```bash
   python evaluation/run_downstream_eval.py --data-config configs/data.yaml --config configs/classifier.yaml --synthetic-dir outputs/cyclegan/synthetic_kl3
   ```
5. **Run unit tests:**
   ```bash
   pytest -q
   ```
