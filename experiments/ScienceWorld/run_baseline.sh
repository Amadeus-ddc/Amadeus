#!/usr/bin/env bash
# ScienceWorld Streaming — Baseline (no memory / ReAct)
# Run from: /data/hzy/Amadeus/amadeus/experiments/ScienceWorld
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/data/hzy/miniconda3/envs/amadeus1/bin/python"
MODEL="/data/hzy/models/Qwen2.5-7B-Instruct"
MODEL_NAME="${MODEL}"
PORT=8101

echo "=== ScienceWorld Streaming — Baseline (none) ==="
date

# Start vLLM server
CUDA_VISIBLE_DEVICES=5 ${PYTHON} -m vllm.entrypoints.openai.api_server \
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
${PYTHON} run_sciworld_streaming.py \
    --method none \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:${PORT}/v1" \
    --api_key "token-abc123" \
    --max_steps 150

EXIT_CODE=$?
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

echo "Done. Exit code: ${EXIT_CODE}"
date
exit ${EXIT_CODE}
