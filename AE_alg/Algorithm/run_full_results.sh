#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

: "${DRAFT_MODEL:?Set DRAFT_MODEL to the local Qwen2.5-0.5B-Instruct checkpoint}"
: "${PRM_MODEL:?Set PRM_MODEL to the local Qwen2.5-Math-PRM-7B checkpoint}"
: "${CONFIG1_TARGET_MODEL:?Set CONFIG1_TARGET_MODEL to the local 7B target checkpoint}"
: "${CONFIG2_TARGET_MODEL:?Set CONFIG2_TARGET_MODEL to the local 1.5B target checkpoint}"

export DATASETS="math500 gsm8k gaokao2023en olympiadbench"
export SUBSET_SIZE="0"
export RUN_TAG="${RUN_TAG:-table3-full-results}"

run_config() {
    local config_name="$1"
    export TARGET_MODEL="$2"

    bash common/run_local.sh "${config_name}" rsd table3_src custom
    bash common/run_local.sh "${config_name}" brsd table3_src custom
    bash common/run_local.sh "${config_name}" brsd_optimized table3_src custom
}

echo "Running the complete Table 3 experiment matrix."
echo "Results tag: ${RUN_TAG}"
run_config Config1 "${CONFIG1_TARGET_MODEL}"
run_config Config2 "${CONFIG2_TARGET_MODEL}"
echo "Table 3 full reproduction complete."
