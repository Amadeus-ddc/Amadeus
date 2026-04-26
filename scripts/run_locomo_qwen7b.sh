#!/usr/bin/env bash
# ===========================================================================
# LoCoMo 评测 - Qwen2.5-7B-Instruct
# 基于官方 scripts/run_locomo_muti.sh
# 环境: amadeus1
# 依赖: 需要先启动 vLLM 服务 (bash scripts/start_vllm_qwen.sh)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-$(command -v /data/hzy/miniconda3/envs/amadeus1/bin/python 2>/dev/null || command -v python)}"

RUN_SCRIPT="${ROOT_DIR}/experiments/LoCoMo/run_locomo.py"
DEFAULT_DATA_FILE="${ROOT_DIR}/dataset/LoCoMo/locomo10.json"
if [[ ! -f "${DEFAULT_DATA_FILE}" && -f "${ROOT_DIR}/../amadeus/dataset/LoCoMo/locomo10.json" ]]; then
    DEFAULT_DATA_FILE="${ROOT_DIR}/../amadeus/dataset/LoCoMo/locomo10.json"
fi
DATA_FILE="${DATA_FILE:-${DEFAULT_DATA_FILE}}"
EMBED_MODEL="${EMBED_MODEL:-${ROOT_DIR}/models/all-MiniLM-L6-v2}"

LOG_BASE_DIR="${ROOT_DIR}/experiments/LoCoMo/logs"
TS="$(date +%Y%m%d_%H%M%S)"

MODEL_NAME="${MODEL_NAME:-qwen2.5-7b-instruct}"

# Judge model
JUDGE_MODEL="${JUDGE_MODEL:-qwen2.5-32b-instruct}"
JUDGE_API_BASE="${JUDGE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-sk-f418eb8b1eb941a7975419cceb23bc89}"

ABLATION_MODE="${ABLATION_MODE:-adaptive_buffer_fixed_sp}"
FIXED_SP_COUNT="${FIXED_SP_COUNT:-3}"

declare -a JOBS=(
  "GPU=6 SAMPLE_IDS=conv-26,conv-30,conv-41,conv-42,conv-43,conv-44,conv-47,conv-48,conv-49,conv-50 OUT=locomo_qwen7b_${TS} MAX_WORKERS=3 PORT=8006 MAIN_API_BASE=http://localhost:8006/v1 MAIN_API_KEY=local"
)

echo "============================================"
echo "  LoCoMo × Qwen2.5-7B-Instruct"
echo "  Ablation: ${ABLATION_MODE}"
echo "============================================"

for job in "${JOBS[@]}"; do
    eval "$job"

    : "${PORT:?PORT not set in job entry}"
    IFS=',' read -ra SAMPLE_ARR <<< "${SAMPLE_IDS}"
    for SAMPLE in "${SAMPLE_ARR[@]}"; do
      SAMPLE_TRIMMED="$(echo "${SAMPLE}" | xargs)"

      TARGET_RUN_DIR="${LOG_BASE_DIR}/${OUT}"
      mkdir -p "${TARGET_RUN_DIR}"

      LOG_FILE="${TARGET_RUN_DIR}/console_${SAMPLE_TRIMMED}.log"

      ${PYTHON_BIN} "${RUN_SCRIPT}" \
        --data_file "${DATA_FILE}" \
        --sample_id "${SAMPLE_TRIMMED}" \
        --model_name "${MODEL_NAME}" \
        --api_base "${MAIN_API_BASE}" \
        --api_key "${MAIN_API_KEY}" \
        --judge_model_name "${JUDGE_MODEL}" \
        --judge_api_base "${JUDGE_API_BASE}" \
        --judge_api_key "${JUDGE_API_KEY}" \
        --ablation_mode "${ABLATION_MODE}" \
        --fixed_sp_count "${FIXED_SP_COUNT}" \
        --embedding_model "${EMBED_MODEL}" \
        --run_name "${OUT}" \
        --max_workers "${MAX_WORKERS}" \
        > "${LOG_FILE}" 2>&1 &
      echo "Launched -> sample ${SAMPLE_TRIMMED}, log ${LOG_FILE}"
    done
done

echo "Waiting for all jobs to complete..."
wait

# Token stats analysis
echo "Calculating Memory Graph Token Stats..."
for job in "${JOBS[@]}"; do
    eval "$job"
    ${PYTHON_BIN} "${ROOT_DIR}/experiments/LoCoMo/analyze_memory_tokens.py" --run_dirs "${LOG_BASE_DIR}/${OUT}"
done

echo "All jobs completed."
