#!/bin/bash
# LongMemEval evaluation pipeline for Qwen2.5-7B-Instruct
# Uses the original repo scripts with minimal modifications
# Usage: bash run.sh [full-history|no-retrieval|both|eval-only]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/repo"
DATA_DIR="/data/hzy/Amadeus/amadeus/dataset/longmemeval"
LOG_DIR="${SCRIPT_DIR}/logs"

# Dataset file
IN_FILE="${DATA_DIR}/longmemeval_s_cleaned.json"
REF_FILE="${DATA_DIR}/longmemeval_oracle.json"

# Offline mode for tokenizer loading
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/data/hzy/Amadeus/amadeus/data/hf_cache/"

MODE=${1:-"both"}
mkdir -p "${LOG_DIR}"

# ============ Generation (uses repo/src/generation/run_generation.sh) ============
run_gen() {
    local retriever_alias=$1   # full-history-session | no-retrieval
    local reading_method=$2    # con | direct | con-separate
    local topk=${3:-1000}
    local history_format=${4:-"json"}
    local useronly=${5:-"false"}

    echo "========================================"
    echo "Generation: model=qwen2.5-7b-instruct retriever=${retriever_alias} reading=${reading_method}"
    echo "========================================"

    cd "${REPO_DIR}/src/generation"
    bash run_generation.sh \
        "${IN_FILE}" \
        "qwen2.5-7b-instruct" \
        "${retriever_alias}" \
        "${topk}" \
        "${history_format}" \
        "${useronly}" \
        "${reading_method}" \
        2>&1 | tee "${LOG_DIR}/gen_${retriever_alias}_${reading_method}_$(date +%Y%m%d_%H%M%S).log"
    cd "${SCRIPT_DIR}"
}

# ============ Evaluation (uses repo/src/evaluation/evaluate_qa.py) ============
run_eval() {
    local hyp_file=$1
    echo "========================================"
    echo "Evaluating: $(basename ${hyp_file})"
    echo "Judge: qwen2.5-32b-instruct"
    echo "========================================"

    cd "${REPO_DIR}/src/evaluation"
    python evaluate_qa.py \
        "qwen2.5-32b-instruct" \
        "${hyp_file}" \
        "${REF_FILE}" \
        2>&1 | tee "${hyp_file}.eval_log"
    cd "${SCRIPT_DIR}"
}

# ============ Main Pipeline ============
if [[ "${MODE}" == "full-history" || "${MODE}" == "both" ]]; then
    echo ">>> Stage 1: Full History Session (CoT)"
    run_gen "full-history-session" "con" 1000
fi

if [[ "${MODE}" == "no-retrieval" || "${MODE}" == "both" ]]; then
    echo ">>> Stage 2: No Retrieval (CoT)"
    run_gen "no-retrieval" "con" 1000
fi

if [[ "${MODE}" == "eval-only" ]]; then
    echo ">>> Evaluating existing results..."
fi

# Auto-evaluate all un-evaluated hypothesis files
echo ""
echo ">>> Evaluation with Qwen2.5-32B judge..."
GEN_LOG_DIR="${REPO_DIR}/generation_logs"
for hyp_file in $(find "${GEN_LOG_DIR}" -name "*_testlog_*" -not -name "*.eval*" 2>/dev/null | sort -r); do
    if [[ ! -f "${hyp_file}.eval-results-qwen2.5-32b-instruct" ]]; then
        run_eval "${hyp_file}"
    else
        echo "Already evaluated: ${hyp_file}"
    fi
done

# Print metrics
echo ""
echo ">>> Printing metrics..."
cd "${REPO_DIR}/src/evaluation"
for eval_file in $(find "${GEN_LOG_DIR}" -name "*.eval-results-qwen2.5-32b-instruct" 2>/dev/null | sort -r); do
    echo "--- ${eval_file} ---"
    python print_qa_metrics.py "${eval_file}" 2>&1
done

echo ""
echo ">>> Pipeline complete!"
