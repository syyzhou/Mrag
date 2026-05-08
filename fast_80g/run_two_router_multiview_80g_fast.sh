#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-./Qwen2.5-3B}"
TRAIN_FILE="${TRAIN_FILE:-./datasets/nyc/preprocessed/train_dst_time_layered12_far3_mismatch3_tail2_ep200.json}"
TEST_FILE="${TEST_FILE:-./datasets/nyc/preprocessed/test_dst_time_layered12_far3_mismatch3_tail2_ep200.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./fast_80g/output/two-router-target-poi-multiview-dst-time-layered12-far3-mismatch3-tail2-ep200-80g-fast}"
SEQ_LEN="${SEQ_LEN:-8192}"
DATASET_NAME="${DATASET_NAME:-nyc}"
THRESHOLD_MIB="${THRESHOLD_MIB:-10000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-./fast_80g/logs/run_two_router_multiview_80g_fast}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./fast_80g/ds_stage2_80g_fast.json}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/run_status_${RUN_TS}.log"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
TF32_ENABLED="${TF32_ENABLED:-True}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-2}"

TRAIN_TRAJ_EMB="${TRAIN_TRAJ_EMB:-./datasets/nyc/preprocessed/trajectory_embeddings.pt}"
TEST_TRAJ_EMB="${TEST_TRAJ_EMB:-./datasets/nyc/preprocessed/test_embeddings.pt}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

find_gpu_over_threshold() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$(timestamp)] nvidia-smi not found; cannot detect GPU availability." | tee -a "${STATUS_LOG}" >&2
    return 1
  fi

  nvidia-smi --query-gpu=index,memory.total,memory.free,name --format=csv,noheader,nounits \
  | awk -F', ' -v threshold="${THRESHOLD_MIB}" '$3 >= threshold {print $1","$2","$3","$4}' \
  | sort -t',' -k3,3nr \
  | head -n 1
}

wait_for_gpu() {
  local stage="$1"
  echo "[$(timestamp)] Waiting for GPU for ${stage} with free memory >= ${THRESHOLD_MIB} MiB" | tee -a "${STATUS_LOG}" >&2
  while true; do
    local gpu_info
    if ! gpu_info="$(find_gpu_over_threshold 2>/dev/null)"; then
      echo "[$(timestamp)] GPU detection failed for ${stage}; please check NVIDIA driver/CUDA visibility." | tee -a "${STATUS_LOG}" >&2
      return 1
    fi
    if [[ -n "${gpu_info}" ]]; then
      echo "${gpu_info}"
      return 0
    fi
    echo "[$(timestamp)] No suitable GPU for ${stage}; sleeping ${POLL_SECONDS}s" | tee -a "${STATUS_LOG}" >&2
    sleep "${POLL_SECONDS}"
  done
}

echo "[$(timestamp)] Model path: ${MODEL_PATH}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Train file: ${TRAIN_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Test file: ${TEST_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Output dir: ${OUTPUT_DIR}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] DeepSpeed config: ${DEEPSPEED_CONFIG}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Train batch: ${TRAIN_BATCH_SIZE}, grad accumulation: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a "${STATUS_LOG}"

GPU_INFO="$(wait_for_gpu training)"
GPU_ID="$(echo "$GPU_INFO" | cut -d',' -f1)"
GPU_TOTAL="$(echo "$GPU_INFO" | cut -d',' -f2)"
GPU_FREE="$(echo "$GPU_INFO" | cut -d',' -f3)"
GPU_NAME="$(echo "$GPU_INFO" | cut -d',' -f4-)"

echo "[$(timestamp)] Using GPU ${GPU_ID} (${GPU_NAME}) for training (total=${GPU_TOTAL} MiB, free=${GPU_FREE} MiB)" | tee -a "${STATUS_LOG}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

TRAIN_CMD=(
  python fast_80g/supervised-fine-tune-qlora-two-router-80g.py
  --model_name_or_path "${MODEL_PATH}"
  --bf16 True
  --output_dir "${OUTPUT_DIR}"
  --use_flash_attn False
  --dataset "${TRAIN_FILE}"
  --dataset_name "${DATASET_NAME}"
  --low_rank_training True
  --num_train_epochs 1
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --evaluation_strategy no
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --save_strategy no
  --save_steps 100
  --learning_rate 2e-5
  --weight_decay 0.0
  --warmup_steps 20
  --lr_scheduler_type constant_with_warmup
  --logging_steps 1
  --logging_dir "${LOG_DIR}/trainer"
  --deepspeed "${DEEPSPEED_CONFIG}"
  --model_max_length "${SEQ_LEN}"
  --tf32 "${TF32_ENABLED}"
  --gradient_checkpointing False
  --use_trajectory_routing True
  --trajectory_fusion_mode gate
  --share_traj_projector False
  --router1_use_shared_expert True
  --router1_shared_expert_weight 1.0
)

if [[ -n "${TRAIN_TRAJ_EMB}" ]]; then
  TRAIN_CMD+=(--trajectory_embedding_path "${TRAIN_TRAJ_EMB}")
fi

echo "[$(timestamp)] Starting training..." | tee -a "${STATUS_LOG}"
"${TRAIN_CMD[@]}"
echo "[$(timestamp)] Training finished." | tee -a "${STATUS_LOG}"

GPU_INFO_AFTER="$(wait_for_gpu evaluation)"
GPU_ID_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f1)"
GPU_TOTAL_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f2)"
GPU_FREE_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f3)"
GPU_NAME_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f4-)"

echo "[$(timestamp)] Starting evaluation on GPU ${GPU_ID_AFTER} (${GPU_NAME_AFTER}) (total=${GPU_TOTAL_AFTER} MiB, free=${GPU_FREE_AFTER} MiB)" | tee -a "${STATUS_LOG}"
export CUDA_VISIBLE_DEVICES="${GPU_ID_AFTER}"

EVAL_CMD=(
  python eval_two_router.py
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --test_file "${TEST_FILE}"
  --dataset_name "${DATASET_NAME}"
  --context_size "${SEQ_LEN}"
  --seq_len "${SEQ_LEN}"
  --batch_size "${GEN_BATCH_SIZE}"
)

if [[ "${EVAL_DEVICE}" != "auto" ]]; then
  EVAL_CMD+=(--device "${EVAL_DEVICE}")
fi

if [[ -n "${TEST_TRAJ_EMB}" ]]; then
  EVAL_CMD+=(--trajectory_embedding_path "${TEST_TRAJ_EMB}")
fi

"${EVAL_CMD[@]}"
echo "[$(timestamp)] Evaluation finished." | tee -a "${STATUS_LOG}"
