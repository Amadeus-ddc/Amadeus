#!/bin/bash
# ===========================================================================
# GPQA ReMem Streaming Evaluation — Qwen2.5-7B-Instruct
#
# ReMem = ExpRAG + Refine Memory (post-answer pruning)
# 答题 prompt 和 ExpRAG 完全一致，区别只在 Evolve 阶段剪枝记忆
#
# 前置: 先启动 vLLM server
#   cd experiments/GPQA && bash run_eval.sh serve
#
# 用法:
#   bash scripts/run_remem_qwen7b.sh [dataset] [max_examples]
#   bash scripts/run_remem_qwen7b.sh diamond        # 完整 198 题
#   bash scripts/run_remem_qwen7b.sh diamond 5      # 快速测试 5 题
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPQA_DIR="${ROOT_DIR}/experiments/GPQA"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

API_PORT="${API_PORT:-8003}"
TOP_K="${TOP_K:-4}"
DATASET="${1:-diamond}"
MAX_EXAMPLES="${2:-}"

DATA_FILE="${ROOT_DIR}/dataset/GPQA/gpqa_${DATASET}.csv"
OUTPUT_DIR="${GPQA_DIR}/logs/results_evo"

echo "============================================"
echo "  GPQA ReMem Streaming Evaluation"
echo "  Model:     Qwen2.5-7B-Instruct"
echo "  API:       http://localhost:${API_PORT}/v1"
echo "  Top-K:     ${TOP_K}"
echo "  Dataset:   gpqa_${DATASET}"
echo "============================================"

EXTRA_ARGS=""
if [ -n "${MAX_EXAMPLES}" ]; then
    EXTRA_ARGS="--max_examples ${MAX_EXAMPLES}"
    echo "  Max examples: ${MAX_EXAMPLES} (debug mode)"
fi

# Offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${ROOT_DIR}/data/hf_cache"

cd "${GPQA_DIR}"
${PYTHON_BIN} -m evo_memory.run_evo_eval \
    --agent remem \
    --model_name qwen2.5-7b-instruct \
    --api_base "http://localhost:${API_PORT}/v1" \
    --api_key EMPTY \
    --data_path "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --top_k "${TOP_K}" \
    --run_tag "remem_qwen7b" \
    --seed 0 \
    --verbose \
    ${EXTRA_ARGS} \
    "$@"

echo ""
echo "============================================"
echo "  Done. Results: ${OUTPUT_DIR}"
echo "============================================"
