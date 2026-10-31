# PIRA: Personalized Inter-Rater Agreement for Multi-Annotator Medical Segmentation

Official implementation of **PIRA** — a two-stage probabilistic UNet that learns rater-specific segmentation by routing latent samples from a shared prior through AdaSineLoRA-modulated expert projection heads.

---

## Model Overview

PIRA operates in two stages:
- **Stage 1** — trains the probabilistic UNet prior (backbone) to model the joint distribution of all rater annotations using SIGReg regularisation.
- **Stage 2** — freezes the backbone and trains rater-specific `ExpertProjectionHeads` (with AdaSineLoRA query modulation) to route latent samples to each rater's prediction.

---

## Repository Structure

```
PIRA_github/
├── code/                     # Core training and evaluation code
│   ├── train_lidc.py         # LIDC training (stage 1 & 2)
│   ├── train_npc.py          # NPC training (stage 1 & 2)
│   ├── evaluate_lidc.py      # LIDC evaluation
│   ├── evaluate_npc.py       # NPC evaluation
│   ├── configs/              # Dataset config files
│   ├── dataloader/           # Data loading utilities
│   ├── lib/                  # Model definitions (PIRA, AdaSineLoRA, SIGReg)
│   └── utils/                # Training utilities
├── weights/
│   ├── lidc/                 # LIDC weights — all 4 folds, stage 2
│   └── npc/
│       └── stage2/           # NPC stage-2 weights (alpha=0.0, tau=4, lam=1.0)
├── visualize/
│   ├── visualize_lidc_all.sh          # Run all LIDC visualizations at once
│   ├── make_diversity_panel_lidc.py   # Latent diversity figure (LIDC)
│   ├── make_diversity_panel_npc.py    # Latent diversity figure (NPC)
│   └── jim_vis/              # Interpretability experiments (9 scripts)
│       ├── 00_baseline_eval.py
│       ├── 01_attribution_study.py
│       ├── 02_prior_coverage.py
│       ├── 03_latent_traversal.py
│       ├── 04_sine_coefficients.py
│       ├── 05_sine_ablation.py
│       ├── 06_effective_rank.py
│       ├── 07_uncertainty_calibration.py
│       ├── 08_diversity_collapse.py
│       └── run_all.sh
└── scripts/
    ├── train_lidc.sh          # End-to-end LIDC training
    ├── train_npc.sh           # End-to-end NPC training
    ├── evaluate_lidc.sh       # LIDC evaluation wrapper
    └── evaluate_npc.sh        # NPC evaluation wrapper
```

---

## Environment

```bash
conda create -n dpersona python=3.9
conda activate dpersona
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn matplotlib opencv-python tqdm h5py tensorboard captum
```

---

## Datasets

Both datasets are bundled locally under `dataset/`.

| Dataset | Path | Format |
|---------|------|--------|
| LIDC-IDRI | `dataset/lidc/lidc_attributes.pkl` (set via `DATA_PATH` in `configs/params_lidc.yaml`) | `.pkl` (pre-processed patches) |
| NPC (MMIS2024TASK1) | `dataset/npc/MMIS2024TASK1/` (set via `DATA_PATH` in `configs/params_npc.yaml`) | `.h5` volumes |

---

## Evaluation with Provided Weights

### LIDC (4-fold cross-validation)

```bash
cd code

# Stage 2 (full model), K=50 samples
python evaluate_lidc.py --stage 2 --gpu 0 --test_num 50 --save_path ../weights/lidc/
```

### NPC

```bash
cd code

# Stage 2
python evaluate_npc.py --stage 2 --gpu 0 --test_num 50 \
    --save_path ../weights/npc/stage2/
```

Or use the wrapper scripts:
```bash
./scripts/evaluate_lidc.sh 2 0 50   # stage=2, fold=0, K=50
./scripts/evaluate_npc.sh  2   50   # stage=2, K=50
```

---

## Training from Scratch

### LIDC

```bash
./scripts/train_lidc.sh 0 0   # fold=0, GPU=0
```

Key stage-1 hyperparameters: `sigreg_lambda=0.01`  
Key stage-2 hyperparameters: defaults (`tau=2, lam=3.0, alpha=0.5, sigreg_lambda=0.01`)

### NPC

```bash
./scripts/train_npc.sh 0   # GPU=0
```

Key stage-1 hyperparameters: `sigreg_lambda=0.05, beta=0.5`  
Key stage-2 hyperparameters: `loss_tau=4, loss_lam=1.0, loss_alpha=0.0, sigreg_lambda=0.0`


    --ckpt     ../weights/npc/stage2/PIRA_MMIS2024TASK1_best.pth \
    --out_dir  ../outputs/figures/
```


