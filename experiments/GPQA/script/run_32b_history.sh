#!/bin/bash
# ===========================================================================
# GPQA 32B History Baseline
# Uses DashScope API with Qwen2.5-32B-Instruct
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPQA_DIR="$(dirname "${SCRIPT_DIR}")"
ROOT_DIR="$(cd "${GPQA_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

# Load API credentials
source "${GPQA_DIR}/../.env"
API_BASE="${OPENAI_API_BASE}"
API_KEY="${OPENAI_API_KEY}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-32b-instruct}"

DATA_FILE="${ROOT_DIR}/dataset/GPQA/gpqa_diamond.csv"
OUTPUT_DIR="${GPQA_DIR}/logs/results_evo"

echo "============================================"
echo "  GPQA 32B History Baseline"
echo "  Model:   ${MODEL_NAME}"
echo "  API:     ${API_BASE}"
echo "  Dataset: gpqa_diamond"
echo "============================================"

cd "${GPQA_DIR}"
${PYTHON_BIN} -m evo_memory.run_evo_eval \
    --agent history \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --data_path "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_tag "history_${MODEL_NAME}" \
    --seed 0 \
    --verbose \
    "$@"

echo ""
echo "  Done! Results in: ${OUTPUT_DIR}"
