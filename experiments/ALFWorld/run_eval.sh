#!/bin/bash
# ===========================================================================
# ALFWorld Evaluation — method-agnostic
#
# Supports both STREAMING (memory evolves across episodes) and
# BATCH (non-streaming, no cross-episode memory) modes.
#
# Usage:
#   bash run_eval.sh streaming [--method amadeus] [extra args]
#   bash run_eval.sh batch     [--method none]    [extra args]
#   bash run_eval.sh serve     # Start vLLM server
#   bash run_eval.sh all       # Start server + run streaming eval
#
# Environment variables:
#   GPU_ID=0                  # GPU for vLLM server (default: 0)
#   API_PORT=8000             # vLLM port (default: 8000)
#   METHOD=none               # Memory method (default: none)
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL="${MODEL:-${ROOT_DIR}/models/Qwen2.5-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-7b-instruct}"
PORT="${API_PORT:-8000}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-${ROOT_DIR}/models/all-MiniLM-L6-v2}"
METHOD="${METHOD:-none}"
PYTHON="${PYTHON:-$(command -v /data/hzy/miniconda3/envs/amadeus1/bin/python 2>/dev/null || command -v python)}"

serve() {
    echo "Starting vLLM server with model: ${MODEL} on GPU: ${GPU_ID}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON}" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port ${PORT} \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 8192 \
        --trust-remote-code \
        --dtype auto
}

wait_for_server() {
    echo "Waiting for vLLM server to be ready..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not start within 4 minutes"
    return 1
}

streaming() {
    echo "============================================"
    echo "  ALFWorld STREAMING Evaluation"
    echo "  Method:  ${METHOD}"
    echo "  Model:   ${MODEL_NAME}"
    echo "  API:     http://localhost:${PORT}/v1"
    echo "============================================"
    "${PYTHON}" "${SCRIPT_DIR}/run_alfworld_streaming.py" \
        --method "${METHOD}" \
        --model_name "${MODEL_NAME}" \
        --api_base "http://localhost:${PORT}/v1" \
        --embedding_model "${EMBEDDING_MODEL}" \
        "$@"
}

batch() {
    echo "============================================"
    echo "  ALFWorld BATCH (non-streaming) Evaluation"
    echo "  Method:  ${METHOD}"
    echo "  Model:   ${MODEL_NAME}"
    echo "  API:     http://localhost:${PORT}/v1"
    echo "============================================"
    "${PYTHON}" "${SCRIPT_DIR}/run_alfworld_batch.py" \
        --method "${METHOD}" \
        --model_name "${MODEL_NAME}" \
        --api_base "http://localhost:${PORT}/v1" \
        --embedding_model "${EMBEDDING_MODEL}" \
        "$@"
}

case "${1}" in
    serve)
        serve
        ;;
    streaming)
        shift
        streaming "$@"
        ;;
    batch)
        shift
        batch "$@"
        ;;
    all)
        shift
        serve &
        SERVER_PID=$!
        echo "vLLM server PID: ${SERVER_PID}"
        wait_for_server || { kill ${SERVER_PID} 2>/dev/null; exit 1; }
        streaming "$@"
        EVAL_EXIT=$?
        echo "Stopping vLLM server..."
        kill ${SERVER_PID} 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
        exit ${EVAL_EXIT}
        ;;
    *)
        echo "Usage: $0 {serve|streaming|batch|all} [extra args]"
        echo ""
        echo "Examples:"
        echo "  $0 serve                                    # Start vLLM server"
        echo "  $0 streaming --method amadeus               # Streaming with Amadeus memory"
        echo "  $0 streaming --method none                  # Streaming baseline (no memory)"
        echo "  $0 batch --method none                      # Batch baseline"
        echo "  $0 batch --method amadeus                   # Batch with per-episode memory"
        echo "  METHOD=amadeus $0 all                       # Server + streaming eval"
        echo ""
        echo "Methods:  none, amadeus"
        echo "Env vars: GPU_ID, API_PORT, METHOD"
        exit 1
        ;;
esac
