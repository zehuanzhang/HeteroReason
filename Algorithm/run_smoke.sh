#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

: "${DRAFT_MODEL:?Set DRAFT_MODEL to the local Qwen2.5-0.5B-Instruct checkpoint}"
: "${PRM_MODEL:?Set PRM_MODEL to the local Qwen2.5-Math-PRM-7B checkpoint}"
if [[ -z "${CONFIG1_TARGET_MODEL:-${TARGET_MODEL:-}}" ]]; then
    echo "Set CONFIG1_TARGET_MODEL (or TARGET_MODEL) to the local 7B target checkpoint" >&2
    exit 2
fi

export DATASETS="math500"
export SUBSET_SIZE="8"
export RUN_TAG="${RUN_TAG:-table3-smoke}"
export TARGET_MODEL="${CONFIG1_TARGET_MODEL:-${TARGET_MODEL:-}}"

echo "Running the Table 3 Config1 smoke test on 8 fixed Math500 examples."
bash common/run_local.sh Config1 rsd table3_src custom
bash common/run_local.sh Config1 brsd table3_src custom
bash common/run_local.sh Config1 brsd_optimized table3_src custom
echo "Table 3 smoke test complete."
