#!/bin/bash
# 批量运行所有 Locomo 样本 (使用 Qwen)

# 确保 vLLM 服务已经启动

python amadeus/experiments/run_locomo.py \
    --model_name qwen2.5-14b-instruct \
    --api_base http://localhost:8000/v1 \
    --api_key EMPTY \
    --sample_id all
