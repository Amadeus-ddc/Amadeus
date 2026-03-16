#!/bin/bash
# ===========================================================================
# GPQA 32B ExpRAG
# Uses DashScope API with Qwen2.5-32B-Instruct + embedding retrieval
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
TOP_K="${TOP_K:-4}"

DATA_FILE="${ROOT_DIR}/dataset/GPQA/gpqa_diamond.csv"
OUTPUT_DIR="${GPQA_DIR}/logs/results_evo"

echo "============================================"
echo "  GPQA 32B ExpRAG"
echo "  Model:   ${MODEL_NAME}"
echo "  API:     ${API_BASE}"
echo "  Top-K:   ${TOP_K}"
echo "  Dataset: gpqa_diamond"
echo "============================================"

cd "${GPQA_DIR}"
${PYTHON_BIN} -m evo_memory.run_evo_eval \
    --agent exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --data_path "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --top_k "${TOP_K}" \
    --run_tag "exprag_${MODEL_NAME}_topk${TOP_K}" \
    --seed 0 \
    --verbose \
    "$@"

echo ""
echo "  Done! Results in: ${OUTPUT_DIR}"
