#!/usr/bin/env bash
# ============================================================
# PIRA — NPC training (stage 1 then stage 2)
# Usage: ./scripts/train_npc.sh [GPU]
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CODE="${REPO}/code"
CONDA_ENV="${CONDA_ENV:-dpersona}"
GPU="${1:-0}"
OUT="${REPO}/outputs/npc"

mkdir -p "${OUT}/stage1" "${OUT}/stage2"

echo "=== NPC Stage 1 ==="
cd "${CODE}"
conda run -n "${CONDA_ENV}" python train_npc.py \
    --stage 1 \
    --epochs 150 \
    --val_num 10 \
    --gpu "${GPU}" \
    --sigreg_lambda 0.05 \
    --beta 0.5 \
    --prior_sample_num 10 \
    --save_path "${OUT}/stage1/"

echo "=== NPC Stage 2 ==="
conda run -n "${CONDA_ENV}" python train_npc.py \
    --stage 2 \
    --epochs 150 \
    --val_num 10 \
    --gpu "${GPU}" \
    --stage1_ckpt_dir "${OUT}/stage1" \
    --loss_tau 4 \
    --loss_lam 1.0 \
    --loss_alpha 0.0 \
    --loss_rho 1.0 \
    --loss_delta 0.0 \
    --sigreg_lambda 0.0 \
    --save_path "${OUT}/stage2/"

echo "[done] outputs in ${OUT}/"
