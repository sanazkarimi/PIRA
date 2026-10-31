#!/usr/bin/env bash
# ============================================================
# PIRA — LIDC training (stage 1 then stage 2)
# Usage: ./scripts/train_lidc.sh [FOLD] [GPU]
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CODE="${REPO}/code"
CONDA_ENV="${CONDA_ENV:-dpersona}"
FOLD="${1:-0}"
GPU="${2:-0}"
OUT="${REPO}/outputs/lidc"

mkdir -p "${OUT}/stage1_fold${FOLD}" "${OUT}/stage2_fold${FOLD}"

echo "=== LIDC Stage 1 (fold ${FOLD}) ==="
cd "${CODE}"
conda run -n "${CONDA_ENV}" python train_lidc.py \
    --stage 1 \
    --epochs 150 \
    --val_num 10 \
    --gpu "${GPU}" \
    --sigreg_lambda 0.01 \
    --save_path "${OUT}/stage1_fold${FOLD}/"

echo "=== LIDC Stage 2 (fold ${FOLD}) ==="
conda run -n "${CONDA_ENV}" python train_lidc.py \
    --stage 2 \
    --epochs 150 \
    --val_num 10 \
    --gpu "${GPU}" \
    --stage1_ckpt_dir "${OUT}/stage1_fold${FOLD}" \
    --loss_tau 2 \
    --loss_lam 3.0 \
    --loss_alpha 0.5 \
    --loss_rho 1.0 \
    --loss_delta 0.0 \
    --sigreg_lambda 0.01 \
    --save_path "${OUT}/stage2_fold${FOLD}/"

echo "[done] outputs in ${OUT}/"
