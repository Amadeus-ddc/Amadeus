#!/bin/bash

# 设置项目根目录 (绝对路径)
REPO_ROOT="/home/ubuntu/hzy/crl/Amadeus"
cd $REPO_ROOT

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$REPO_ROOT/amadeus/experiments/log"
mkdir -p $LOG_DIR

# 实验日志 (Python logging 输出)
EXP_LOG="$LOG_DIR/locomo_batch_qwen_${TIMESTAMP}.log"
# 终端输出 (Stdout/Stderr)
NOHUP_LOG="$LOG_DIR/nohup_qwen_${TIMESTAMP}.out"

echo "Starting batch experiment in background..."
echo "Experiment Log: $EXP_LOG"
echo "Nohup Output: $NOHUP_LOG"

# 使用 nohup 后台运行
nohup python amadeus/experiments/run_locomo.py \
    --model_name qwen2.5-7b-instruct \
    --api_base http://localhost:8000/v1 \
    --api_key EMPTY \
    --sample_id all \
    --log_path "$EXP_LOG" > "$NOHUP_LOG" 2>&1 &

echo "Process ID: $!"
