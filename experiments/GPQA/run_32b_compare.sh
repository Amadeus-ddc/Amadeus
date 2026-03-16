#!/bin/bash
# ===========================================================================
# GPQA 32B Comparison: NoMemory vs History
# Uses DashScope API with Qwen2.5-32B-Instruct
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

# Load API credentials
source "${SCRIPT_DIR}/../.env"
API_BASE="${OPENAI_API_BASE}"
API_KEY="${OPENAI_API_KEY}"
MODEL_NAME="qwen2.5-32b-instruct"

DATA_FILE="$(cd "${SCRIPT_DIR}/../.." && pwd)/dataset/GPQA/gpqa_diamond.csv"
OUTPUT_DIR="${SCRIPT_DIR}/logs/results_evo"

echo "============================================"
echo "  GPQA 32B Comparison Experiments"
echo "  Model:   ${MODEL_NAME}"
echo "  API:     ${API_BASE}"
echo "  Dataset: gpqa_diamond"
echo "============================================"

# --- Run 1: NoMemory baseline ---
echo ""
echo ">>> [1/2] Running NoMemory baseline..."
cd "${SCRIPT_DIR}"
${PYTHON_BIN} -m evo_memory.run_evo_eval \
    --agent none \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --data_path "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_tag no_memory_32b \
    --seed 0 \
    --verbose

# --- Run 2: History baseline ---
echo ""
echo ">>> [2/2] Running History baseline..."
${PYTHON_BIN} -m evo_memory.run_evo_eval \
    --agent history \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --data_path "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_tag history_32b \
    --seed 0 \
    --verbose

echo ""
echo "============================================"
echo "  Both experiments completed!"
echo "  Results in: ${OUTPUT_DIR}"
echo "============================================"
