#!/bin/bash
#SBATCH --job-name=alfworld-streaming-none
#SBATCH --output=/data/user/zhu851/tzx/amadeus/scripts/sbatch/alfworld/streaming_none_%j.log
#SBATCH --error=/data/user/zhu851/tzx/amadeus/scripts/sbatch/alfworld/streaming_none_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# ===========================================================================
# ALFWorld Streaming + No Memory (baseline)
# max_steps=50
# Metrics: SR, PR
# ===========================================================================
set -e

export ALFWORLD_DATA="/data/user/zhu851/tzx/amadeus/dataset/ALFWorld"

SCRIPT_DIR="/data/user/zhu851/tzx/amadeus/experiments/ALFWorld"
MODEL="/data/user/zhu851/models/Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b-instruct"
PORT=8010
CONDA_ENV="amadeus1"

echo "=== ALFWorld Streaming + No Memory ==="
echo "Job ID: ${SLURM_JOB_ID}"
date

source /data/user/zhu851/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

python -m vllm.entrypoints.openai.api_server \
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
python run_alfworld_streaming.py \
    --method none \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:${PORT}/v1" \
    --api_key "token-abc123" \
    --max_steps 50

EXIT_CODE=$?
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

echo "Done. Exit code: ${EXIT_CODE}"
date
exit ${EXIT_CODE}
