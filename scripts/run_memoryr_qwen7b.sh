#!/bin/bash
# ===========================================================================
# Memory-R 评测 - Qwen2.5-7B-Instruct
# 基于官方 evaluation/sh/eval.sh
# 环境: amadeus1
# 注意: 此脚本直接加载模型到 GPU（vllm 离线推理），需要空闲 GPU
# ===========================================================================
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MEMORYR_DIR="${ROOT_DIR}/experiments/Memory-R"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

# HuggingFace offline mode (for HPC nodes without internet)
export HF_HOME="${ROOT_DIR}/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_NAME_OR_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
PROMPT_TYPE="${1:-qwen25-math-cot}"
GPU="${GPU:-6}"
OUTPUT_DIR="${MODEL_NAME_OR_PATH}/math_eval"
SPLIT="test"
NUM_TEST_SAMPLE=-1

export CUDA_VISIBLE_DEVICES=${GPU}

echo "============================================"
echo "  Memory-R eval × Qwen2.5-7B-Instruct"
echo "  Model: ${MODEL_NAME_OR_PATH}"
echo "  Prompt: ${PROMPT_TYPE}"
echo "  GPU: ${GPU}"
echo "============================================"

cd "${MEMORYR_DIR}/evaluation"

# English open datasets (gsm8k as primary benchmark)
DATA_NAME="gsm8k"
TOKENIZERS_PARALLELISM=false \
${PYTHON_BIN} -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

# English multiple-choice datasets
DATA_NAME="aqua,sat_math,mmlu_stem"
TOKENIZERS_PARALLELISM=false \
${PYTHON_BIN} -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --num_shots 5
