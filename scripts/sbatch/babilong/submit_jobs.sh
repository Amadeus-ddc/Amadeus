#!/bin/bash
# Submit BABILong experiments to HPC

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_DIR="${SCRIPT_DIR}/sbatch/babilong"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
  lightmem [SPLITS] [TASKS]     Run LightMemory experiments
  amadeus                        Run Amadeus experiments
  baseline                       Run baseline (no memory) experiments
  all                           Run all three experiments

Examples:
  $0 lightmem                    # Default: 1k,4k,16k splits
  $0 lightmem 16k               # Only 16k split
  $0 lightmem 0k,1k,2k,4k,8k,16k,32k,64k,128k  # All splits
  $0 lightmem 4k qa1,qa2        # 4k split, qa1-qa2 tasks
  $0 amadeus                     # Run Amadeus
  $0 baseline                    # Run baseline
  $0 all                         # Run all three

EOF
}

submit_lightmem() {
    local splits="${1:-1k,4k,16k}"
    local tasks="${2:-qa1,qa2,qa3,qa4,qa5}"

    echo -e "${YELLOW}Submitting LightMemory job...${NC}"
    echo "  Splits: $splits"
    echo "  Tasks: $tasks"

    JOB_ID=$(sbatch --export=SPLITS=$splits,TASKS=$tasks "${SBATCH_DIR}/lightmem.sh" | awk '{print $NF}')
    echo -e "${GREEN}✓ Submitted with Job ID: $JOB_ID${NC}"
    echo "  Monitor: tail -f /data/user/zhu851/tzx/amadeus/slurm_logs/${JOB_ID}_lightmem_babilong.out"
}

submit_amadeus() {
    echo -e "${YELLOW}Submitting Amadeus job...${NC}"
    JOB_ID=$(sbatch "${SBATCH_DIR}/amadeus.sh" | awk '{print $NF}')
    echo -e "${GREEN}✓ Submitted with Job ID: $JOB_ID${NC}"
    echo "  Monitor: tail -f /data/user/zhu851/tzx/amadeus/scripts/sbatch/babilong/amadeus_${JOB_ID}.log"
}

submit_baseline() {
    echo -e "${YELLOW}Submitting baseline job...${NC}"
    JOB_ID=$(sbatch "${SBATCH_DIR}/none.sh" | awk '{print $NF}')
    echo -e "${GREEN}✓ Submitted with Job ID: $JOB_ID${NC}"
    echo "  Monitor: tail -f /data/user/zhu851/tzx/amadeus/scripts/sbatch/babilong/none_${JOB_ID}.log"
}

if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

case "$1" in
    lightmem)
        submit_lightmem "$2" "$3"
        ;;
    amadeus)
        submit_amadeus
        ;;
    baseline)
        submit_baseline
        ;;
    all)
        submit_lightmem "$2" "$3"
        submit_amadeus
        submit_baseline
        ;;
    -h|--help)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        print_usage
        exit 1
        ;;
esac
