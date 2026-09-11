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

export DATASETS="${DATASETS:-math500}"
export SUBSET_SIZE="0"
export RUN_TAG="${RUN_TAG:-table3-key-results}"
export TARGET_MODEL="${CONFIG1_TARGET_MODEL:-${TARGET_MODEL:-}}"
KEY_METHODS="${KEY_METHODS:-brsd_optimized}"

read -r -a METHOD_ARRAY <<< "$KEY_METHODS"

echo "============================================================"
echo " HeteroReason Table 3 representative reproduction"
echo " Config1 / complete Math500 / ${KEY_METHODS}"
echo " Default method: brsd_optimized (BRSD + optimizations)"
echo " Default historical runtime: 106 min 38 s on 3x RTX 3090 GPUs"
echo " To run one method at a time:"
echo "   KEY_METHODS=\"rsd\" bash run_key_results.sh"
echo "   KEY_METHODS=\"brsd\" bash run_key_results.sh"
echo "   KEY_METHODS=\"brsd_optimized\" bash run_key_results.sh"
echo " Results tag: ${RUN_TAG}"
echo "============================================================"

for METHOD in "${METHOD_ARRAY[@]}"; do
    bash common/run_local.sh Config1 "$METHOD" table3_src custom
done

echo ""
echo "Key-result reproduction complete."
echo "Metrics: results/Config1/<method>/${RUN_TAG}/math500/"
echo "Logs:    logs/Config1/<method>/${RUN_TAG}/math500.log"
