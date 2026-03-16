#!/usr/bin/env bash
# ===========================================================================
# BABILong 评测 - Qwen2.5-7B-Instruct
# 环境: amadeus1
# 依赖: 需要先启动 vLLM 服务 (bash scripts/start_vllm_qwen.sh)
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BABILONG_DIR="${ROOT_DIR}/experiments/babilong"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

# HuggingFace offline mode
export HF_HOME="${ROOT_DIR}/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RESULTS_FOLDER="${RESULTS_FOLDER:-${BABILONG_DIR}/logs/babilong_evals}"
DATASET_NAME="RMT-team/babilong-1k-samples"
MODEL_NAME="qwen2.5-7b-instruct"
TOKENIZER="/data/hzy/models/Qwen2.5-7B-Instruct"
API_URL="${API_URL:-http://localhost:8006/v1/completions}"

TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")

echo "============================================"
echo "  BABILong × Qwen2.5-7B-Instruct"
echo "  API: ${API_URL}"
echo "  Tasks: ${TASKS[*]}"
echo "  Lengths: ${LENGTHS[*]}"
echo "  Output: ${RESULTS_FOLDER}"
echo "============================================"

${PYTHON_BIN} "${BABILONG_DIR}/scripts/run_model_on_babilong.py" \
    --results_folder "${RESULTS_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --model_name "${MODEL_NAME}" \
    --tokenizer_path "${TOKENIZER}" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    --system_prompt "You are a helpful assistant." \
    --use_chat_template \
    --use_instruction \
    --use_examples \
    --use_post_prompt \
    --api_url "${API_URL}" \
    "$@"
