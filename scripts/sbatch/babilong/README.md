# BABILong Experiments on HPC

## Overview
This directory contains sbatch scripts for running BABILong experiments on HPC with different memory systems:
- **lightmem.sh**: LightMemory adapter
- **amadeus.sh**: Amadeus memory system (full self-play)
- **none.sh**: Baseline (no memory)

## Dataset
All scripts use the HPC cached dataset at:
```
/data/user/zhu851/hf_cache/datasets/RMT-team___babilong
```

Available splits: `0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k`
Available tasks: `qa1, qa2, qa3, qa4, qa5`

## LightMemory (lightmem.sh)

### Default Configuration
- Splits: `1k,4k,16k`
- Tasks: `qa1,qa2,qa3,qa4,qa5`
- Output: `/data/user/zhu851/tzx/amadeus/results/lightmem_babilong`

### Submit Job
```bash
# Default (1k, 4k, 16k splits)
sbatch /data/hzy/Amadeus/amadeus/scripts/sbatch/babilong/lightmem.sh

# Custom splits
sbatch --export=SPLITS=1k /data/hzy/Amadeus/amadeus/scripts/sbatch/babilong/lightmem.sh
sbatch --export=SPLITS=0k,1k,2k,4k,8k,16k,32k,64k,128k /data/hzy/Amadeus/amadeus/scripts/sbatch/babilong/lightmem.sh

# Custom tasks
sbatch --export=TASKS=qa1,qa2 /data/hzy/Amadeus/amadeus/scripts/sbatch/babilong/lightmem.sh

# Custom output directory
sbatch --export=OUTPUT_DIR=/data/user/zhu851/tzx/amadeus/results/lightmem_custom /data/hzy/Amadeus/amadeus/scripts/sbatch/babilong/lightmem.sh

# Combine multiple exports
sbatch --export=SPLITS=4k,16k,128k,TASKS=qa1,qa2,qa3,OUTPUT_DIR=/data/user/zhu851/tzx/amadeus/results/lightmem_test /data/hzy/Amadeus/amadeus/scripts/sbatch/babilong/lightmem.sh
```

## Amadeus (amadeus.sh)

### Submit Job
```bash
sbatch /data/hzy/Amadeus/amadeus/scripts/sbatch/babilong/amadeus.sh
```

### Configuration
Edit the script directly to modify:
- `--lengths`: Dataset splits (default: 0k 1k 2k 4k 8k 16k)
- `--tasks`: Tasks to run (default: qa1 qa2 qa3 qa4 qa5)
- `--ablation_mode`: Memory mode (default: adaptive_buffer_fixed_sp)

## Baseline (none.sh)

### Submit Job
```bash
sbatch /data/hzy/Amadeus/amadeus/scripts/sbatch/babilong/none.sh
```

### Configuration
Edit the script directly to modify:
- `--lengths`: Dataset splits (default: 0k 1k 2k 4k 8k 16k)
- `--tasks`: Tasks to run (default: qa1 qa2 qa3 qa4 qa5)

## Monitoring Jobs

### Check job status
```bash
squeue -u zhu851
```

### View job output
```bash
tail -f /data/user/zhu851/tzx/amadeus/slurm_logs/JOBID_lightmem_babilong.out
```

### Cancel job
```bash
scancel JOBID
```

## Results

Results are saved to the output directory specified in each script:
- LightMemory: `/data/user/zhu851/tzx/amadeus/results/lightmem_babilong/`
- Amadeus: `/data/user/zhu851/tzx/amadeus/experiments/babilong/babilong_evals_amadeus/`
- Baseline: `/data/user/zhu851/tzx/amadeus/experiments/babilong/babilong_evals/`

Each result directory contains:
- `results_*.csv`: Results in CSV format
- `results_*.json`: Results in JSON format with detailed metrics

## Notes

1. **Dataset Loading**: The scripts automatically load from the HPC cache via the `HPC_DATASET_PATH` environment variable
2. **GPU Memory**: Adjust `--mem` in sbatch directives if needed (default: 80G for LightMemory, 32G for others)
3. **Time Limit**: Adjust `--time` based on dataset size (default: 24h for LightMemory, 48h for Amadeus, 24h for baseline)
4. **Partition**: Using `acd_u` partition (adjust if needed)
