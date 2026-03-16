#!/bin/bash
# ===========================================================================
# ScienceWorld 评测 - Qwen2.5-7B-Instruct
# 环境: amadeus1
# 依赖: 需要先启动 vLLM 服务 (bash scripts/start_vllm_qwen.sh)
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCIWORLD_DIR="${ROOT_DIR}/experiments/ScienceWorld"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

MODEL="qwen2.5-7b-instruct"
API_BASE="${API_BASE:-http://localhost:8006/v1}"
OUTPUT_DIR="${SCIWORLD_DIR}/logs/results"

echo "============================================"
echo "  ScienceWorld × Qwen2.5-7B-Instruct"
echo "  API: ${API_BASE}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

${PYTHON_BIN} "${SCIWORLD_DIR}/eval_sciworld.py" \
    --model "${MODEL}" \
    --api-base "${API_BASE}" \
    --temperature 0.4 \
    --max-history 2 \
    --output-dir "${OUTPUT_DIR}" \
    --seed 0 \
    "$@"
