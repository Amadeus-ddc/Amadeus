#!/bin/bash
# ===========================================================================
# MATH-500 Evo-Memory Streaming Evaluation — local run script
#
# Uses DashScope API (no local vLLM needed).
# Logs output to experiments/MATH/logs/results_evo/
#
# Usage:
#   bash run_evo_eval.sh none        # Run bare model
#   bash run_evo_eval.sh history     # Run full history
#   bash run_evo_eval.sh exprag      # Run ExpRAG
#   bash run_evo_eval.sh all         # Run all three sequentially
#   bash run_evo_eval.sh none 10     # Quick test (10 problems)
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATH_DIR="${SCRIPT_DIR}"
ROOT_DIR="$(cd "${MATH_DIR}/../.." && pwd)"

# Offline mode
export HF_HOME="${ROOT_DIR}/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Load API credentials
source "${MATH_DIR}/../.env"
API_BASE="${OPENAI_API_BASE}"
API_KEY="${OPENAI_API_KEY}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-32b-instruct}"
TOP_K="${TOP_K:-4}"

DATA_PATH="${ROOT_DIR}/dataset/MATH/HuggingFaceH4___math-500"
OUTPUT_DIR="${MATH_DIR}/logs/results_evo"
EMBEDDER_PATH="${ROOT_DIR}/models/all-MiniLM-L6-v2"

MAX_EXAMPLES=""
if [ -n "$2" ]; then
    MAX_EXAMPLES="--max_examples $2"
fi

run_agent() {
    local AGENT=$1
    echo "============================================"
    echo "  MATH-500 Evo-Memory Evaluation"
    echo "  Agent:   ${AGENT}"
    echo "  Model:   ${MODEL_NAME}"
    echo "  API:     ${API_BASE}"
    echo "============================================"

    cd "${MATH_DIR}"
    python -m evo_memory.run_evo_eval \
        --agent "${AGENT}" \
        --model_name "${MODEL_NAME}" \
        --api_base "${API_BASE}" \
        --api_key "${API_KEY}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --embedder_path "${EMBEDDER_PATH}" \
        --top_k "${TOP_K}" \
        --run_tag "${AGENT}_${MODEL_NAME}" \
        --seed 0 \
        --verbose \
        ${MAX_EXAMPLES}

    echo ""
    echo "  Done [${AGENT}]! Results in: ${OUTPUT_DIR}"
}

case "${1}" in
    none|history|exprag)
        run_agent "$1"
        ;;
    all)
        run_agent "none"
        run_agent "history"
        run_agent "exprag"
        ;;
    *)
        echo "Usage: $0 {none|history|exprag|all} [max_examples]"
        echo ""
        echo "Examples:"
        echo "  $0 none          # Bare model baseline"
        echo "  $0 history       # Full history injection"
        echo "  $0 exprag        # ExpRAG retrieval"
        echo "  $0 all           # Run all three"
        echo "  $0 none 10       # Quick test (10 problems)"
        exit 1
        ;;
esac
