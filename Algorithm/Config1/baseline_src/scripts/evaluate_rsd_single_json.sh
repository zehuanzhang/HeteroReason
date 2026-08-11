#!/bin/bash

# Script for evaluating RSD with main_online_tree_single.py using different approaches
# Usage: ./scripts/evaluate_rsd_single.sh [--subset SIZE] [--approach APPROACH] [--mode MODE] [--nsys]
# APPROACH can be:
#   notree   - Direct target model (no beam search)
#   beam4    - Beam search: width=1, expansion=4, regular scores
#   beam8    - Beam search: width=1, expansion=8, regular scores
#   beam44   - Beam search: width=4, expansion=4, regular scores
#   beam4_log, beam8_log, beam44_log - Same configs using log scores
#   Additional beam variants: beam16, bbeam*, ebbeam*, dbeam*, etc.
#   
#   Hierarchical speculative decoding (0.5B -> 1.5B -> 7B):
#   hbeam*   - Hierarchical spec decoding + beam search
#   hbbeam*  - Hierarchical spec decoding + backtracking beam search
#   hbbeam*_prefetch - Same with target prefetch
#   hbbeam*_prefetch_cache - Same with target prefetch + cache
# MODE can be: local, server

# Default values
SUBSET_SIZE=10
APPROACH="beam4"  # Default to beam search with width=1, expansion=4
MODE="server"    # Default to server mode
SEED=43
DATA_NAME="math500" #,gsm8k,gaokao2023en,olympiadbench"
#DRAFT_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-Math-1.5B-Instruct-W8A8"
# DRAFT_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-0.5B-Instruct"
DRAFT_MODEL="/mnt/ccnas2/bdp/zz3822/SpeculativeDecoding/robin/HuggingFaceModel/Qwen2.5_0.5B_Instruct_quantized_w8a8"
#TARGET_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-0.5B-Instruct"

#DRAFT_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-Math-1.5B-Instruct" #"/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-1.5B-Instruct"
#TARGET_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-Math-7B-Instruct" #"/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-7B-Instruct"
#TARGET_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-Math-1.5B-Instruct"

#TARGET_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-1.5B-Instruct"
#DRAFT_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-1.5B-Instruct"

# PRM_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Skywork-o1-Open-PRM-Qwen-2.5-1.5B" #"/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-Math-PRM-7B"
# DRAFT_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-1.5B-Instruct"
TARGET_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-7B-Instruct"
PRM_MODEL="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-Math-PRM-7B"
PRM_THRESHOLD=0.7
NSYS=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --subset)
      SUBSET_SIZE="$2"
      shift 2
      ;;
    --approach)
      APPROACH="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --data)
      DATA_NAME="$2"
      shift 2
      ;;
    --nsys)
      NSYS=1
      shift 1
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=== Running RSD evaluation (main_online_tree_single.py) ==="
echo "Approach: $APPROACH"
echo "Mode: $MODE"
echo "Dataset: $DATA_NAME"
echo "Subset size: $SUBSET_SIZE"
echo "Seed: $SEED"
echo "NSYS profiling: $NSYS"

# Base command with mode-specific settings
if [ "$MODE" = "local" ]; then
    BASE_CMD="python main_online_tree_single.py \
        --data_names $DATA_NAME \
        --fixed_subset_size $SUBSET_SIZE \
        --draft_model_name_or_path $DRAFT_MODEL \
        --target_model_name_or_path $TARGET_MODEL \
        --prm_name_or_path $PRM_MODEL \
        --use_local_models \
        --output_dir \"outputs/${APPROACH}_single_local\" \
        --prompt_type \"qwen25-math-cot\" \
        --apply_chat_template \
        --prm_threshold $PRM_THRESHOLD \
        --max_steps 100 \
        --patience 5 \
        --seed $SEED \
        --save_outputs"
         
else
    BASE_CMD="python main_online_tree_single.py \
        --data_names $DATA_NAME \
        --fixed_subset_size $SUBSET_SIZE \
        --draft_model_name_or_path $DRAFT_MODEL \
        --draft_model_ip_address \"http://localhost:12340/v1\" \
        --target_model_name_or_path $TARGET_MODEL \
        --target_model_ip_address \"http://localhost:12341/v1\" \
        --prm_name_or_path $PRM_MODEL \
        --prm_ip_address \"http://localhost:12342/v1\" \
        --output_dir \"outputs/${APPROACH}_single_server\" \
        --prompt_type \"qwen25-math-cot\" \
        --apply_chat_template \
        --prm_threshold $PRM_THRESHOLD \
        --max_steps 100 \
        --patience 5 \
        --seed $SEED \
        --save_outputs"
fi

# Optionally wrap with nsys profile
if [ "$NSYS" = "1" ]; then
    mkdir -p outputs
    NSYS_CMD="nsys profile --trace=cuda,nvtx,osrt --trace-fork-before-exec=true --kill=sigterm --output=outputs/nsys_report_${APPROACH}_single_${MODE}_$(date +%Y%m%d_%H%M%S)"
else
    NSYS_CMD=""
fi

# Run the appropriate approach
case $APPROACH in
  notree)
    echo "Running with direct target model (no tree/beam)"
    eval "$NSYS_CMD $BASE_CMD"
    ;;
  beam1)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0"
    ;;
  beam1_prefetch)
    echo "Running with beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_target_prefetch"
    ;;
  beam2)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2"
    ;;
  beam2_prefetch)
    echo "Running with beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_target_prefetch"
    ;;
  beam4)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4"
    ;;
  beam4_prefetch)
    echo "Running with beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_target_prefetch"
    ;;
  beam8)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8"
    ;;
  beam8_prefetch)
    echo "Running with beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_target_prefetch"
    ;;
  beam16)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16"
    ;;
  beam16_prefetch)
    echo "Running with beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_target_prefetch"
    ;;
  bbeam1)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_backtracking"
    ;;
  bbeam1_prefetch)
    echo "Running with backtracking beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_backtracking --enable_target_prefetch"
    ;;
  bbeam1_prefetch_cache)
    echo "Running with backtracking beam search + target prefetch (cached partials)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache"
    ;;
  bbeam2)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_backtracking"
    ;;
  bbeam2_prefetch)
    echo "Running with backtracking beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_backtracking --enable_target_prefetch"
    ;;
  bbeam2_prefetch_cache)
    echo "Running with backtracking beam search + target prefetch (cached partials)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache"
    ;;
  bbeam4)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_backtracking"
    ;;
  bbeam4_prefetch)
    echo "Running with backtracking beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_backtracking --enable_target_prefetch"
    ;;
  bbeam4_prefetch_cache)
    echo "Running with backtracking beam search + target prefetch (cached partials)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache"
    ;;
  bbeam8)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_backtracking"
    ;;
  bbeam8_prefetch)
    echo "Running with backtracking beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_backtracking --enable_target_prefetch"
    ;;
  bbeam8_prefetch_cache)
    echo "Running with backtracking beam search + target prefetch (cached partials)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache"
    ;;
  bbeam16)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_backtracking"
    ;;
  bbeam16_prefetch)
    echo "Running with backtracking beam search + target prefetch"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_backtracking --enable_target_prefetch"
    ;;
  bbeam16_prefetch_cache)
    echo "Running with backtracking beam search + target prefetch (cached partials)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache"
    ;;
  ebbeam1)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_backtracking --backtrack_temperature 0.7 --backtrack_expansion_factor 16"
    ;;
  ebbeam2)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_backtracking --backtrack_expansion_factor 16"
    ;;
  ebbeam4)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_backtracking --backtrack_expansion_factor 16"
    ;;
  ebbeam8)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_backtracking --backtrack_expansion_factor 16"
    ;;
  ebbeam16)
    echo "Running with beam search"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_backtracking --backtrack_expansion_factor 16"
    ;;
  # Hierarchical speculative decoding approaches (hbbeam* variants)
  # Uses 0.5B token drafter -> 1.5B step drafter -> 7B target with PRM
  hbbeam1)
    echo "Running with hierarchical spec decoding + backtracking beam search (expansion=1)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_backtracking --enable_hierarchical_spec"
    ;;
  hbbeam1_prefetch)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch (expansion=1)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_backtracking --enable_target_prefetch --enable_hierarchical_spec"
    ;;
  hbbeam1_prefetch_cache)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch cache (expansion=1)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache --enable_hierarchical_spec"
    ;;
  hbbeam2)
    echo "Running with hierarchical spec decoding + backtracking beam search (expansion=2)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_backtracking --enable_hierarchical_spec"
    ;;
  hbbeam2_prefetch)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch (expansion=2)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_backtracking --enable_target_prefetch --enable_hierarchical_spec"
    ;;
  hbbeam2_prefetch_cache)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch cache (expansion=2)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache --enable_hierarchical_spec"
    ;;
  hbbeam4)
    echo "Running with hierarchical spec decoding + backtracking beam search (expansion=4)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_backtracking --enable_hierarchical_spec"
    ;;
  hbbeam4_prefetch)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch (expansion=4)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_backtracking --enable_target_prefetch --enable_hierarchical_spec"
    ;;
  hbbeam4_prefetch_cache)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch cache (expansion=4)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache --enable_hierarchical_spec"
    ;;
  hbbeam8)
    echo "Running with hierarchical spec decoding + backtracking beam search (expansion=8)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_backtracking --enable_hierarchical_spec"
    ;;
  hbbeam8_prefetch)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch (expansion=8)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_backtracking --enable_target_prefetch --enable_hierarchical_spec"
    ;;
  hbbeam8_prefetch_cache)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch cache (expansion=8)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache --enable_hierarchical_spec"
    ;;
  hbbeam16)
    echo "Running with hierarchical spec decoding + backtracking beam search (expansion=16)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_backtracking --enable_hierarchical_spec"
    ;;
  hbbeam16_prefetch)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch (expansion=16)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_backtracking --enable_target_prefetch --enable_hierarchical_spec"
    ;;
  hbbeam16_prefetch_cache)
    echo "Running with hierarchical spec decoding + backtracking beam search + prefetch cache (expansion=16)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_backtracking --enable_target_prefetch --enable_prefetch_cache --enable_hierarchical_spec"
    ;;
  # Non-backtracking hierarchical variants (hbeam*)
  hbeam1)
    echo "Running with hierarchical spec decoding + beam search (expansion=1)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_hierarchical_spec"
    ;;
  hbeam1_prefetch)
    echo "Running with hierarchical spec decoding + beam search + prefetch (expansion=1)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 1 --search_temperature 0.0 --enable_hierarchical_spec --enable_target_prefetch"
    ;;
  hbeam2)
    echo "Running with hierarchical spec decoding + beam search (expansion=2)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_hierarchical_spec"
    ;;
  hbeam2_prefetch)
    echo "Running with hierarchical spec decoding + beam search + prefetch (expansion=2)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 2 --enable_hierarchical_spec --enable_target_prefetch"
    ;;
  hbeam4)
    echo "Running with hierarchical spec decoding + beam search (expansion=4)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_hierarchical_spec"
    ;;
  hbeam4_prefetch)
    echo "Running with hierarchical spec decoding + beam search + prefetch (expansion=4)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 4 --enable_hierarchical_spec --enable_target_prefetch"
    ;;
  hbeam8)
    echo "Running with hierarchical spec decoding + beam search (expansion=8)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_hierarchical_spec"
    ;;
  hbeam8_prefetch)
    echo "Running with hierarchical spec decoding + beam search + prefetch (expansion=8)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 8 --enable_hierarchical_spec --enable_target_prefetch"
    ;;
  hbeam16)
    echo "Running with hierarchical spec decoding + beam search (expansion=16)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_hierarchical_spec"
    ;;
  hbeam16_prefetch)
    echo "Running with hierarchical spec decoding + beam search + prefetch (expansion=16)"
    eval "$NSYS_CMD $BASE_CMD --beam_search --beam_width 1 --beam_expansion_factor 16 --enable_hierarchical_spec --enable_target_prefetch"
    ;;
  *)
    echo "Unknown approach: $APPROACH"
    echo "Valid approaches: notree, beam*, bbeam*, ebbeam*, hbeam*, hbbeam*"
    echo "  hbeam* - hierarchical spec decoding + beam search"
    echo "  hbbeam* - hierarchical spec decoding + backtracking beam search"
    echo "  *_prefetch - with target prefetch"
    echo "  *_prefetch_cache - with target prefetch + cache"
    echo "Valid modes: local, server"
    exit 1
    ;;
esac

echo "Evaluation complete"
