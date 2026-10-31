#!/usr/bin/env bash
# ============================================================
# PIRA — LIDC evaluation
# Usage: ./scripts/evaluate_lidc.sh [STAGE] [FOLD] [TEST_NUM] [GPU]
# ============================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CODE="${REPO}/code"
CONDA_ENV="${CONDA_ENV:-dpersona}"
STAGE="${1:-2}"
FOLD="${2:-0}"
TEST_NUM="${3:-50}"
GPU="${4:-0}"

echo "=== LIDC Evaluation — stage ${STAGE}, fold ${FOLD}"
cd "${CODE}"
conda run -n "${CONDA_ENV}" python evaluate_lidc.py \
    --stage "${STAGE}" \
    --gpu "${GPU}" \
    --test_num "${TEST_NUM}" \
    --save_path "${REPO}/weights/lidc/"
echo "[done] results written to ${REPO}/weights/lidc/performance.txt"
