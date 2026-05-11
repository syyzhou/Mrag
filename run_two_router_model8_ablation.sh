#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="./Qwen2.5-3B"
SEQ_LEN=4096
DATASET_NAME="${DATASET_NAME:-nyc}"
ABLATION_MODE="${ABLATION_MODE:-baseline}"  # baseline | no_shared_router1 | no_router1

TRAIN_FILE="${TRAIN_FILE:-./datasets/${DATASET_NAME}/preprocessed/train_qa_pairs_kqt.json}"
TEST_FILE="${TEST_FILE:-./datasets/${DATASET_NAME}/preprocessed/test_qa_pairs_kqt.txt}"

RUN_TAG_BASE="model8_${DATASET_NAME}_${ABLATION_MODE}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/two-router-${RUN_TAG_BASE}}"

THRESHOLD_MIB="${THRESHOLD_MIB:-20000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-./logs/run_two_router_model8_ablation}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/run_status_${RUN_TS}.log"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
TF32_ENABLED="${TF32_ENABLED:-False}"

ROUTER_MODEL_FILE="model8"
TRAIN_TRAJ_EMB="${TRAIN_TRAJ_EMB:-./datasets/${DATASET_NAME}/preprocessed/trajectory_embeddings.pt}"
TEST_TRAJ_EMB="${TEST_TRAJ_EMB:-./datasets/${DATASET_NAME}/preprocessed/test_embeddings.pt}"

mkdir -p "${LOG_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

find_gpu_over_threshold() {
  nvidia-smi --query-gpu=index,memory.total,memory.free,name --format=csv,noheader,nounits \
  | awk -F', ' -v threshold="${THRESHOLD_MIB}" '$3 >= threshold {print $1","$2","$3","$4}' \
  | sort -t',' -k3,3nr \
  | head -n 1
}

wait_for_gpu() {
  local stage="$1"
  echo "[$(timestamp)] Waiting for GPU for ${stage} (free >= ${THRESHOLD_MIB} MiB)" | tee -a "${STATUS_LOG}" >&2
  while true; do
    local gpu_info
    gpu_info="$(find_gpu_over_threshold || true)"
    if [[ -n "${gpu_info}" ]]; then
      echo "${gpu_info}"
      return 0
    fi
    echo "[$(timestamp)] No suitable GPU for ${stage}; sleeping ${POLL_SECONDS}s" | tee -a "${STATUS_LOG}" >&2
    sleep "${POLL_SECONDS}"
  done
}

ROUTER1_USE_SHARED="True"
ROUTER1_SHARED_WEIGHT="1.0"
DISABLE_ROUTER1="False"

case "${ABLATION_MODE}" in
  baseline)
    ;;
  no_shared_router1)
    ROUTER1_USE_SHARED="False"
    ;;
  no_router1)
    ROUTER1_USE_SHARED="False"
    DISABLE_ROUTER1="True"
    ;;
  *)
    echo "[$(timestamp)] ERROR: unsupported ABLATION_MODE=${ABLATION_MODE}" | tee -a "${STATUS_LOG}" >&2
    exit 1
    ;;
esac

echo "[$(timestamp)] Mode=${ABLATION_MODE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Train=${TRAIN_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Test=${TEST_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Output=${OUTPUT_DIR}" | tee -a "${STATUS_LOG}"

GPU_INFO="$(wait_for_gpu training)"
GPU_ID="$(echo "$GPU_INFO" | cut -d',' -f1)"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

TRAIN_CMD=(
  python supervised-fine-tune-qlora-two-router.py
  --model_name_or_path "${MODEL_PATH}"
  --bf16 True
  --output_dir "${OUTPUT_DIR}"
  --use_flash_attn False
  --dataset "${TRAIN_FILE}"
  --dataset_name "${DATASET_NAME}"
  --low_rank_training True
  --num_train_epochs 1
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 2
  --evaluation_strategy no
  --gradient_accumulation_steps 8
  --save_strategy no
  --learning_rate 2e-5
  --weight_decay 0.0
  --warmup_steps 20
  --lr_scheduler_type constant_with_warmup
  --logging_steps 1
  --logging_dir ./logs
  --deepspeed ds_configs/stage2.json
  --model_max_length "${SEQ_LEN}"
  --tf32 "${TF32_ENABLED}"
  --gradient_checkpointing True
  --router_model_file "${ROUTER_MODEL_FILE}"
  --use_trajectory_routing True
  --trajectory_fusion_mode gate
  --share_traj_projector False
  --router1_use_shared_expert "${ROUTER1_USE_SHARED}"
  --router1_shared_expert_weight "${ROUTER1_SHARED_WEIGHT}"
  --disable_router1 "${DISABLE_ROUTER1}"
)

if [[ -n "${TRAIN_TRAJ_EMB}" ]]; then
  TRAIN_CMD+=(--trajectory_embedding_path "${TRAIN_TRAJ_EMB}")
fi

"${TRAIN_CMD[@]}"

GPU_INFO_AFTER="$(wait_for_gpu evaluation)"
GPU_ID_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f1)"
export CUDA_VISIBLE_DEVICES="${GPU_ID_AFTER}"

EVAL_CMD=(
  python eval_two_router.py
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --test_file "${TEST_FILE}"
  --dataset_name "${DATASET_NAME}"
  --context_size "${SEQ_LEN}"
  --seq_len "${SEQ_LEN}"
  --batch_size 1
  --router_model_file "${ROUTER_MODEL_FILE}"
)

if [[ "${EVAL_DEVICE}" != "auto" ]]; then
  EVAL_CMD+=(--device "${EVAL_DEVICE}")
fi
if [[ -n "${TEST_TRAJ_EMB}" ]]; then
  EVAL_CMD+=(--trajectory_embedding_path "${TEST_TRAJ_EMB}")
fi

"${EVAL_CMD[@]}"
