#!/usr/bin/env bash
# ============================================================
# PIRA — NPC evaluation
# Usage: ./scripts/evaluate_npc.sh [STAGE] [TEST_NUM] [GPU]
# ============================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CODE="${REPO}/code"
CONDA_ENV="${CONDA_ENV:-dpersona}"
STAGE="${1:-2}"
TEST_NUM="${2:-50}"
GPU="${3:-0}"

echo "=== NPC Evaluation — stage ${STAGE}"
cd "${CODE}"
conda run -n "${CONDA_ENV}" python evaluate_npc.py \
    --stage "${STAGE}" \
    --gpu "${GPU}" \
    --test_num "${TEST_NUM}" \
    --save_path "${REPO}/weights/npc/stage${STAGE}/"
echo "[done] results written to ${REPO}/weights/npc/stage${STAGE}/performance.txt"
