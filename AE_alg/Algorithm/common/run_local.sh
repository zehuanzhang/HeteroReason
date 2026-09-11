#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <config-name> <rsd|brsd|brsd_optimized|beam1|bbeam1|pipeline> <source-dir> [smoke|full|custom]" >&2
    exit 2
fi

CONFIG_NAME="$1"
METHOD="$2"
SOURCE_DIR="$(cd "$3" && pwd)"
RUN_MODE="${4:-smoke}"
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGORITHM_ROOT="$(cd "$COMMON_DIR/.." && pwd)"
DATA_DIR="$COMMON_DIR/external/qwen25_math_evaluation/data"

: "${DRAFT_MODEL:?Set DRAFT_MODEL to the local 0.5B draft checkpoint}"
: "${TARGET_MODEL:?Set CONFIG1_TARGET_MODEL/CONFIG2_TARGET_MODEL or TARGET_MODEL}"
: "${PRM_MODEL:?Set PRM_MODEL to the local Qwen2.5-Math-PRM-7B checkpoint}"

case "$RUN_MODE" in
    smoke)
        DATASETS="${DATASETS:-math500}"
        SUBSET_SIZE="${SUBSET_SIZE:-8}"
        ;;
    full)
        DATASETS="${DATASETS:-math500 gsm8k gaokao2023en olympiadbench}"
        SUBSET_SIZE="${SUBSET_SIZE:-0}"
        ;;
    custom)
        DATASETS="${DATASETS:-math500}"
        SUBSET_SIZE="${SUBSET_SIZE:-8}"
        ;;
    *)
        echo "Unknown run mode: $RUN_MODE (expected smoke, full, or custom)" >&2
        exit 2
        ;;
esac

case "$METHOD" in
    rsd|beam1)
        METHOD_FLAGS=(
            --beam_search
            --beam_width 1
            --beam_expansion_factor 1
            --search_temperature 0.0
        )
        ;;
    brsd|bbeam1)
        METHOD_FLAGS=(
            --beam_search
            --beam_width 1
            --beam_expansion_factor 1
            --search_temperature 0.0
            --enable_backtracking
        )
        ;;
    brsd_optimized)
        METHOD_FLAGS=(
            --beam_search
            --beam_width 1
            --beam_expansion_factor 1
            --search_temperature 0.0
            --enable_backtracking
            --enable_target_prefetch
            --enable_prefetch_cache
        )
        ;;
    pipeline)
        METHOD_FLAGS=(
            --beam_search
            --beam_width 1
            --beam_expansion_factor 1
            --search_temperature 0.0
            --enable_backtracking
            --enable_target_prefetch
            --enable_prefetch_cache
            --enable_draft_pipeline_prefetch
        )
        ;;
    *)
        echo "Unknown method: $METHOD" >&2
        exit 2
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
PRM_THRESHOLD="${PRM_THRESHOLD:-0.7}"
SEED="${SEED:-43}"
MAX_STEPS="${MAX_STEPS:-100}"
PATIENCE="${PATIENCE:-5}"
RUN_TAG="${RUN_TAG:-$RUN_MODE}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "RUN_TAG may only contain letters, numbers, dot, underscore, and hyphen" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$COMMON_DIR${PYTHONPATH:+:$PYTHONPATH}"

RESULT_ROOT="$ALGORITHM_ROOT/results/$CONFIG_NAME/$METHOD/$RUN_TAG"
LOG_ROOT="$ALGORITHM_ROOT/logs/$CONFIG_NAME/$METHOD/$RUN_TAG"
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

read -r -a DATASET_ARRAY <<< "$DATASETS"
for DATASET in "${DATASET_ARRAY[@]}"; do
    OUTPUT_DIR="$RESULT_ROOT"
    LOG_FILE="$LOG_ROOT/${DATASET}.log"
    mkdir -p "$OUTPUT_DIR/$DATASET"

    echo "Running $CONFIG_NAME/$METHOD on $DATASET (mode=$RUN_MODE, subset=$SUBSET_SIZE)"
    (
        cd "$SOURCE_DIR"
        "$PYTHON_BIN" main_online_tree_single.py \
            --data_names "$DATASET" \
            --data_dir "$DATA_DIR" \
            --fixed_subset_size "$SUBSET_SIZE" \
            --draft_model_name_or_path "$DRAFT_MODEL" \
            --target_model_name_or_path "$TARGET_MODEL" \
            --prm_name_or_path "$PRM_MODEL" \
            --use_local_models \
            --output_dir "$OUTPUT_DIR" \
            --prompt_type qwen25-math-cot \
            --apply_chat_template \
            --prm_threshold "$PRM_THRESHOLD" \
            --max_steps "$MAX_STEPS" \
            --patience "$PATIENCE" \
            --seed "$SEED" \
            --save_outputs \
            "${METHOD_FLAGS[@]}"
    ) 2>&1 | tee "$LOG_FILE"
done
