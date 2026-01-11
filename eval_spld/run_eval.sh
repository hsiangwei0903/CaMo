#!/bin/bash

# MODEL_NAMES=("qwenvl_3b" "SpatialLadder-3B" "SpaceR-3B" "camo-3B")
MODEL_NAMES=("qwenvl_3b")
TASK=("VSI-Bench")
SUPPORTED_TASKS=("VSI-Bench" "SPBench-SI" "SPBench-MV" "SPAR-Bench" "ViewSpatial-Bench" "CV-Bench")
# PROMPT_TYPE="thinking"
PROMPT_TYPE="default"
GPU_IDS="0,1,2,3" 
NUM_PROCESSES="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_names)
      IFS=' ' read -r -a MODEL_NAMES <<< "$2"
      shift 2
      ;;
    --prompt_type)
      PROMPT_TYPE="$2"
      shift 2
      ;;
    --gpu_ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --num_processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --task)
      IFS=' ' read -r -a TASK <<< "$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Map model name to model path
declare -A MODEL_CONFIG_DICT
MODEL_CONFIG_DICT["qwenvl_3b"]="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_CONFIG_DICT["qwenvl_7b"]="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_CONFIG_DICT["camo-3B"]="/gpfs/projects/embodied3d/LLaMA-Factory/saves/qwen2_5vl-3b/full/exp1"
# MODEL_CONFIG_DICT["SpatialLadder-3B"]="hongxingli/SpatialLadder-3B"
# MODEL_CONFIG_DICT["SpaceR-3B"]="RUBBISHLIKE/SpaceR-3B"

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export CUDA_DEVICE_ORDER=PCI_BUS_ID

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  MODEL_CONFIG_PATH=${MODEL_CONFIG_DICT[$MODEL_NAME]}

  for task_name in "${TASK[@]}"; do
    if [[ ! " ${SUPPORTED_TASKS[@]} " =~ " ${task_name} " ]]; then
      echo "Task $task_name is not supported. Skipping..."
      continue
    fi
    echo "Running evaluation: model=$MODEL_NAME, task=$task_name"

    python evaluator.py \
      --gpu_ids "$GPU_IDS" \
      --eval_task "$task_name" \
      --log_dir ./logs/"${MODEL_NAME}_${PROMPT_TYPE}" \
      --model_config "$MODEL_CONFIG_PATH" \
      --num_processes "$NUM_PROCESSES" \
      --prompt_type "$PROMPT_TYPE" \
      --num_frames 32 \
      --max_pixels $((512*28*28)) \
      --min_pixels $((16*28*28)) \
      --batch_size 16
  done
done
