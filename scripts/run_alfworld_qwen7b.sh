#!/usr/bin/env bash
# ===========================================================================
# ALFWorld 评测 - Qwen2.5-7B-Instruct
# 环境: amadeus1
# 依赖: 需要先启动 vLLM 服务 (bash scripts/start_vllm_qwen.sh)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

RUN_SCRIPT="${ROOT_DIR}/experiments/ALFWorld/run_alfworld.py"

# ---- LLM API ----
API_BASE="${API_BASE:-http://localhost:8006/v1}"
API_KEY="${API_KEY:-local}"
MODEL_NAME="qwen2.5-7b-instruct"

# ---- Embedding Model ----
EMBED_MODEL="${EMBED_MODEL:-${ROOT_DIR}/models/all-MiniLM-L6-v2}"
EMBED_FLAG=""
if [[ -d "${EMBED_MODEL}" ]]; then
    EMBED_FLAG="--embedding_model ${EMBED_MODEL}"
fi

# ---- ALFWorld 数据 ----
if [[ -z "${ALFWORLD_DATA:-}" ]]; then
    REPO_DATA="${ROOT_DIR}/dataset/ALFWorld"
    CACHE_DATA="$HOME/.cache/alfworld"
    if [[ -d "${REPO_DATA}/json_2.1.1" ]]; then
        export ALFWORLD_DATA="${REPO_DATA}"
    elif [[ -d "${CACHE_DATA}" ]]; then
        export ALFWORLD_DATA="${CACHE_DATA}"
    else
        echo "[ERROR] ALFWORLD_DATA not set. No data at dataset/ALFWorld/ or ~/.cache/alfworld"
        exit 1
    fi
fi

# ---- Experiment Parameters ----
ENV_NUM="${ENV_NUM:-134}"
MAX_STEPS="${MAX_STEPS:-50}"
TEST_TIMES="${TEST_TIMES:-1}"
SEED="${SEED:-42}"
HISTORY_LEN="${HISTORY_LEN:-5}"

# ---- Output ----
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-alfworld_qwen7b_${TS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/ALFWorld/logs}"

echo "============================================"
echo "  ALFWorld × Qwen2.5-7B-Instruct"
echo "  API: ${API_BASE}"
echo "  Envs: ${ENV_NUM}, Steps: ${MAX_STEPS}"
echo "  Output: ${OUTPUT_DIR}/${RUN_NAME}"
echo "============================================"

${PYTHON_BIN} "${RUN_SCRIPT}" \
    --model_name ${MODEL_NAME} \
    --api_base ${API_BASE} \
    --api_key ${API_KEY} \
    --env_num ${ENV_NUM} \
    --max_steps ${MAX_STEPS} \
    --test_times ${TEST_TIMES} \
    --seed ${SEED} \
    --history_length ${HISTORY_LEN} \
    --output_dir ${OUTPUT_DIR} \
    --run_name ${RUN_NAME} \
    ${EMBED_FLAG} \
    "$@"
