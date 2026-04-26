#!/usr/bin/env bash
# ALFWorld Streaming — History (streaming baseline)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-$(command -v /data/hzy/miniconda3/envs/amadeus1/bin/python 2>/dev/null || command -v python)}"
MODEL="${MODEL:-${ROOT_DIR}/models/Qwen2.5-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-${MODEL}}"
PORT=8100
ALFWORLD_DATA="${HOME}/.cache/alfworld"

echo "=== ALFWorld Streaming — History ==="
date

export ALFWORLD_DATA="${ALFWORLD_DATA}"

# Start vLLM server
CUDA_VISIBLE_DEVICES=4 ${PYTHON} -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype auto &
VLLM_PID=$!

echo "Waiting for vLLM server (port ${PORT})..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 2
done

cd "${SCRIPT_DIR}"
${PYTHON} run_alfworld_streaming.py \
    --method history \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:${PORT}/v1" \
    --api_key "token-abc123" \
    --max_steps 50 \
    --history_window 5 \
    --history_key_steps 5

EXIT_CODE=$?
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

echo "Done. Exit code: ${EXIT_CODE}"
date
exit ${EXIT_CODE}
