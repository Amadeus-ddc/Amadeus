#!/usr/bin/env bash
# ALFWorld Streaming — ExpRAG (3 runs)
# GPU1 上已有 vLLM server (port 8101)，直接复用
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/data/hzy/miniconda3/envs/amadeus1/bin/python"
MODEL_NAME="/data/hzy/models/Qwen2.5-7B-Instruct"
PORT=8101
ALFWORLD_DATA="${HOME}/.cache/alfworld"

export ALFWORLD_DATA="${ALFWORLD_DATA}"

echo "=== ALFWorld Streaming — ExpRAG (3 runs) ==="
date

# 检查 vLLM server 是否可用
if ! curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
    echo "ERROR: vLLM server not found on port ${PORT}"
    exit 1
fi
echo "vLLM server on port ${PORT} is ready."

cd "${SCRIPT_DIR}"

for RUN in 1 2 3; do
    echo ""
    echo "=========================================="
    echo "  Run ${RUN}/3"
    echo "=========================================="
    date

    ${PYTHON} run_alfworld_streaming.py \
        --method exprag \
        --model_name "${MODEL_NAME}" \
        --api_base "http://localhost:${PORT}/v1" \
        --api_key "token-abc123" \
        --max_steps 50 \
        --output_dir "logs/exprag_run${RUN}"

    echo "Run ${RUN} done."
    date
done

echo ""
echo "=== All 3 runs complete. Summaries: ==="
for RUN in 1 2 3; do
    echo "--- Run ${RUN} ---"
    cat "logs/exprag_run${RUN}/summary.json" 2>/dev/null || echo "(no summary)"
done
date
