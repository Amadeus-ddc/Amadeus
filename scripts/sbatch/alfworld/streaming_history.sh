#!/bin/bash
#SBATCH --job-name=alfworld-streaming-history
#SBATCH --output=/data/hzy/Amadeus/amadeus/scripts/sbatch/alfworld/streaming_history_%j.log
#SBATCH --error=/data/hzy/Amadeus/amadeus/scripts/sbatch/alfworld/streaming_history_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# ===========================================================================
# ALFWorld Streaming + History (streaming baseline)
# Injects most recent 5 episodes (task_desc, success, key_steps) into prompt
# max_steps=50
# Metrics: SR per task type + overall
# ===========================================================================
set -e

SCRIPT_DIR="/data/hzy/Amadeus/amadeus/experiments/ALFWorld"
PYTHON="/data/hzy/miniconda3/envs/amadeus1/bin/python"
MODEL="/data/hzy/models/Qwen2.5-7B-Instruct"
PORT=8100
ALFWORLD_DATA="${HOME}/.cache/alfworld"

echo "=== ALFWorld Streaming + History ==="
echo "Job ID: ${SLURM_JOB_ID}"
date

export ALFWORLD_DATA="${ALFWORLD_DATA}"

${PYTHON} -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype auto &
VLLM_PID=$!

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
    --model_name "${MODEL}" \
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
