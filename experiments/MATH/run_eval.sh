#!/bin/bash
# ===========================================================================
# MATH-500 Streaming Evaluation — method-agnostic
#
# Usage:
#   bash run_eval.sh serve                      # Start vLLM server
#   bash run_eval.sh eval --method amadeus      # Run with Amadeus memory
#   bash run_eval.sh eval --method none         # Run baseline (no memory)
#   bash run_eval.sh eval --max_problems 10     # Quick test
#   bash run_eval.sh eval --resume              # Resume interrupted run
#   bash run_eval.sh all --method amadeus       # Start server + run eval
#
# Environment variables:
#   GPU_ID=0                  # GPU for vLLM server (default: 0)
#   API_PORT=8000             # vLLM port (default: 8000)
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# HuggingFace offline mode (for HPC nodes without internet)
export HF_HOME="${ROOT_DIR}/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="/data/hzy/models/Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b-instruct"
PORT="${API_PORT:-8000}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EMBEDDING_MODEL="/data/hzy/Amadeus/amadeus/models/all-MiniLM-L6-v2"

serve() {
    echo "Starting vLLM server with model: ${MODEL} on GPU: ${GPU_ID}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
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

eval_run() {
    echo "============================================"
    echo "  MATH-500 Streaming Evaluation"
    echo "  Model:   ${MODEL_NAME}"
    echo "  API:     http://localhost:${PORT}/v1"
    echo "============================================"
    python "${SCRIPT_DIR}/run_math500.py" \
        --model_name "${MODEL_NAME}" \
        --api_base "http://localhost:${PORT}/v1" \
        --embedding_model "${EMBEDDING_MODEL}" \
        --temperature 0.0 \
        "$@"
}

case "${1}" in
    serve)
        serve
        ;;
    eval)
        shift
        eval_run "$@"
        ;;
    all)
        shift
        serve &
        SERVER_PID=$!
        echo "vLLM server PID: ${SERVER_PID}"
        wait_for_server || { kill ${SERVER_PID} 2>/dev/null; exit 1; }
        eval_run "$@"
        EVAL_EXIT=$?
        echo "Stopping vLLM server..."
        kill ${SERVER_PID} 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
        exit ${EVAL_EXIT}
        ;;
    *)
        echo "Usage: $0 {serve|eval|all} [extra args]"
        echo ""
        echo "Examples:"
        echo "  $0 serve                             # Start vLLM server"
        echo "  $0 eval --method amadeus             # Run with Amadeus memory"
        echo "  $0 eval --method none                # Run baseline (no memory)"
        echo "  $0 eval --max_problems 10            # Quick test (10 problems)"
        echo "  $0 eval --resume                     # Resume interrupted run"
        echo "  $0 all --method amadeus              # Start server + run eval"
        echo ""
        echo "Methods:  none, amadeus"
        echo "Env vars: GPU_ID, API_PORT"
        exit 1
        ;;
esac
