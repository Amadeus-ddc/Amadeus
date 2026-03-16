#!/bin/bash
# ScienceWorld evaluation script for reproducing naive-mode baseline
# Target: Qwen2.5-7B-Instruct, naive mode, average return ~-61.3
#
# Usage:
#   1. Start vLLM server:  bash run_eval.sh serve
#   2. Run evaluation:     bash run_eval.sh eval
#   3. Or do both:         bash run_eval.sh all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="${SCRIPT_DIR}/results"

serve() {
    echo "Starting vLLM server with model: ${MODEL} on GPU: ${GPU_ID}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n sciworld --no-banner \
        python -m vllm.entrypoints.openai.api_server \
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

eval_all() {
    echo "Running ScienceWorld evaluation..."
    conda run -n sciworld --no-banner \
        python "${SCRIPT_DIR}/eval_sciworld.py" \
        --model "${MODEL}" \
        --api-base "http://localhost:${PORT}/v1" \
        --temperature 0.4 \
        --max-history 2 \
        --output-dir "${OUTPUT_DIR}" \
        --seed 0 \
        "$@"
}

case "${1}" in
    serve)
        serve
        ;;
    eval)
        shift
        eval_all "$@"
        ;;
    all)
        shift
        # Start server in background
        serve &
        SERVER_PID=$!
        echo "vLLM server PID: ${SERVER_PID}"

        # Wait for server
        wait_for_server || { kill ${SERVER_PID} 2>/dev/null; exit 1; }

        # Run evaluation
        eval_all "$@"
        EVAL_EXIT=$?

        # Stop server
        echo "Stopping vLLM server..."
        kill ${SERVER_PID} 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null

        exit ${EVAL_EXIT}
        ;;
    *)
        echo "Usage: $0 {serve|eval|all} [extra args for eval]"
        echo ""
        echo "Examples:"
        echo "  $0 serve                    # Start vLLM server"
        echo "  $0 eval                     # Run evaluation (server must be running)"
        echo "  $0 eval --task-ids 0,1,2    # Evaluate specific tasks only"
        echo "  $0 eval --resume            # Resume interrupted evaluation"
        echo "  $0 all                      # Start server + run eval"
        exit 1
        ;;
esac
