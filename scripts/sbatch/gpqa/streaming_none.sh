#!/bin/bash
#SBATCH --job-name=gpqa-streaming-none
#SBATCH --output=/data/user/zhu851/tzx/amadeus/scripts/sbatch/gpqa/streaming_none_%j.log
#SBATCH --error=/data/user/zhu851/tzx/amadeus/scripts/sbatch/gpqa/streaming_none_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ===========================================================================
# GPQA Diamond Streaming + No Memory (baseline)
# Full diamond dataset
# Metrics: Accuracy
# ===========================================================================
set -e

SCRIPT_DIR="/data/user/zhu851/tzx/amadeus/experiments/GPQA"
MODEL="/data/user/zhu851/models/Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b-instruct"
PORT=8013
CONDA_ENV="amadeus1"
PYTHON_BIN="/data/user/zhu851/miniconda3/envs/${CONDA_ENV}/bin/python"

echo "=== GPQA Diamond Streaming + No Memory ==="
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
${PYTHON_BIN} -m evo_memory.run_evo_eval \
    --agent none \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:${PORT}/v1" \
    --api_key EMPTY \
    --data_path "/data/user/zhu851/tzx/amadeus/dataset/GPQA/gpqa_diamond.csv" \
    --output_dir "${SCRIPT_DIR}/logs/results_evo" \
    --seed 0 \
    --top_k 4 \
    --no_self_play \
    --verbose

EXIT_CODE=$?
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

echo "Done. Exit code: ${EXIT_CODE}"
date
exit ${EXIT_CODE}
