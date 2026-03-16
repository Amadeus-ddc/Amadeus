#!/bin/bash
# Evaluate all hypothesis files with Qwen2.5-32B-Instruct judge
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../repo" && pwd)"
DATA_DIR="/data/hzy/Amadeus/amadeus/dataset/longmemeval"
REF_FILE="${DATA_DIR}/longmemeval_oracle.json"
GEN_LOG_DIR="${REPO_DIR}/generation_logs"

cd "${REPO_DIR}/src/evaluation"
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate amadeus1

echo "=== Evaluating all hypothesis files ==="
for hyp_file in $(find "${GEN_LOG_DIR}" -name "*_testlog_*" -not -name "*.eval*" | sort); do
    if [[ -f "${hyp_file}.eval-results-qwen2.5-32b-instruct" ]]; then
        echo "Already evaluated: $(basename ${hyp_file})"
    else
        echo "Evaluating: ${hyp_file}"
        python evaluate_qa.py "qwen2.5-32b-instruct" "${hyp_file}" "${REF_FILE}"
        echo ""
    fi
done

echo ""
echo "=== Printing all metrics ==="
for eval_file in $(find "${GEN_LOG_DIR}" -name "*.eval-results-qwen2.5-32b-instruct" | sort); do
    echo "--- $(basename $(dirname $(dirname $(dirname ${eval_file}))))/$(basename $(dirname $(dirname ${eval_file})))/$(basename $(dirname ${eval_file})) ---"
    python print_qa_metrics.py "${eval_file}"
    echo ""
done
