#!/bin/bash
# Auto-run remaining ALFWorld and ScienceWorld streaming experiments
# Servers: 8101/8103/8105 for ALFWorld, 8102/8104 for ScienceWorld

M=/data/hzy/models/Qwen2.5-7B-Instruct
ALF=/data/hzy/Amadeus/amadeus/experiments/ALFWorld
SCI=/data/hzy/Amadeus/amadeus/experiments/ScienceWorld
PY=/data/hzy/miniconda3/envs/amadeus1/bin/python
LOG=/data/hzy/Amadeus/amadeus/experiments/run_remaining.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

wait_and_run_alf() {
    local port=$1
    local method=$2
    local run=$3
    local outdir="logs/streaming_${method}_7b_run${run}"
    local logfile="${ALF}/logs/streaming_${method}_7b_run${run}.log"
    log "Starting ALFWorld $method run$run on port $port"
    cd $ALF && $PY run_alfworld_streaming.py \
        --method $method --model_name $M \
        --api_base http://localhost:${port}/v1 --api_key token-abc123 \
        --max_steps 50 --output_dir $outdir \
        > $logfile 2>&1
    local sr=$(python3 -c "import json; d=json.load(open('${ALF}/${outdir}/summary.json')); print(d['success_rate'])" 2>/dev/null || echo "N/A")
    log "DONE ALFWorld $method run$run SR=$sr"
}

wait_and_run_sci() {
    local port=$1
    local method=$2
    local run=$3
    local outdir="logs/streaming_${method}_7b_run${run}"
    local logfile="${SCI}/logs/streaming_${method}_7b_run${run}.log"
    log "Starting ScienceWorld $method run$run on port $port"
    cd $SCI && $PY run_sciworld_streaming.py \
        --method $method --model_name $M \
        --api_base http://localhost:${port}/v1 --api_key token-abc123 \
        --output_dir $outdir \
        > $logfile 2>&1
    local sr=$(python3 -c "import json; d=json.load(open('${SCI}/${outdir}/summary.json')); print(d['success_rate'])" 2>/dev/null || echo "N/A")
    log "DONE ScienceWorld $method run$run SR=$sr"
}

log "=== Starting remaining experiments ==="

# Wait for currently running experiments and reuse their servers
# ALFWorld servers: 8101 (none run1), 8103 (exprag run2), 8105 (none run2)
# ScienceWorld servers: 8102 (none run1), 8104 (exprag run2)

# Run in parallel where possible:
# Group 1: wait for run1/run2 to finish, then reuse servers for run3

# ALFWorld: 3 servers available, run none run3 and exprag run3 sequentially on freed servers
(
    # Wait for ALFWorld none run1 to finish (on 8101)
    while ps aux | grep -v grep | grep "run_alfworld_streaming" | grep "8101" > /dev/null 2>&1; do sleep 30; done
    wait_and_run_alf 8101 none 3
) &

(
    # Wait for ALFWorld exprag run2 to finish (on 8103)
    while ps aux | grep -v grep | grep "run_alfworld_streaming" | grep "8103" > /dev/null 2>&1; do sleep 30; done
    wait_and_run_alf 8103 exprag 3
) &

# ScienceWorld: 2 servers, need to run none×2 + exprag×1 more
(
    # Wait for ScienceWorld none run1 to finish (on 8102)
    while ps aux | grep -v grep | grep "run_sciworld_streaming" | grep "8102" > /dev/null 2>&1; do sleep 30; done
    wait_and_run_sci 8102 none 2
    wait_and_run_sci 8102 none 3
) &

(
    # Wait for ScienceWorld exprag run2 to finish (on 8104)
    while ps aux | grep -v grep | grep "run_sciworld_streaming" | grep "8104" > /dev/null 2>&1; do sleep 30; done
    wait_and_run_sci 8104 exprag 3
) &

# Wait for ALFWorld none run2 (on 8105) - no more ALF experiments needed after this
wait

log "=== All remaining experiments done ==="

# Print summary
log "=== FINAL RESULTS ==="
python3 << 'EOF'
import json, os, numpy as np

results = {}
base = "/data/hzy/Amadeus/amadeus/experiments"

configs = [
    ("ALFWorld", "none", f"{base}/ALFWorld/logs/streaming_none_7b"),
    ("ALFWorld", "exprag", f"{base}/ALFWorld/logs/streaming_exprag_7b"),
    ("ScienceWorld", "none", f"{base}/ScienceWorld/logs/streaming_none_7b"),
    ("ScienceWorld", "exprag", f"{base}/ScienceWorld/logs/streaming_exprag_7b"),
]

for dataset, method, prefix in configs:
    srs = []
    for run in ["", "_run1", "_run2", "_run3"]:
        path = f"{prefix}{run}/summary.json"
        if os.path.exists(path):
            d = json.load(open(path))
            srs.append(d["success_rate"])
    if srs:
        print(f"{dataset} {method}: mean={np.mean(srs):.3f} ± {np.std(srs):.3f} (n={len(srs)}, runs={[f'{x:.3f}' for x in srs]})")
EOF
