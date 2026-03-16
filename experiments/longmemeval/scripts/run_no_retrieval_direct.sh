#!/bin/bash
# No-Retrieval + Direct (no CoT, no history)
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../repo" && pwd)"
DATA_DIR="/data/hzy/Amadeus/amadeus/dataset/longmemeval"
IN_FILE="${DATA_DIR}/longmemeval_s_cleaned.json"
LOG_DIR="/data/hzy/Amadeus/amadeus/experiments/longmemeval/logs"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${LOG_DIR}"

OUT_DIR="${REPO_DIR}/generation_logs/no-retrieval/qwen2.5-7b-instruct/direct"
mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}/src/generation"
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate amadeus1

python run_generation.py \
    --in_file "${IN_FILE}" \
    --out_dir "${OUT_DIR}" \
    --model_name "qwen2.5-7b-instruct" \
    --model_alias "qwen2.5-7b-instruct" \
    --retriever_type "no-retrieval" \
    --topk_context 1000 \
    --history_format "json" \
    --useronly "false" \
    --cot "false" \
    --openai_key "sk-62e9728bb07249ec9b5696da52fdc5d2" \
    --openai_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --merge_key_expansion_into_value "none" \
    2>&1 | tee "${LOG_DIR}/no-retrieval_direct_$(date +%Y%m%d_%H%M%S).log"
