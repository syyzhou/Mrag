#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${1:-${DATASET_NAME:-nyc}}"
VARIANT_NAME="${2:-${VARIANT_NAME:-dst_time_layered12_far3_mismatch3_tail2}}"

case "${DATASET_NAME}" in
  nyc|tky|ca) ;;
  *)
    echo "Invalid dataset_name: ${DATASET_NAME}"
    echo "Usage: bash run_single_lora_multiview_shuffle_ablation.sh [nyc|tky|ca] [variant_name]"
    exit 1
    ;;
esac

MODEL_PATH="${MODEL_PATH:-./Qwen2.5-3B}"
TRAIN_FILE="${TRAIN_FILE:-./datasets/${DATASET_NAME}/preprocessed/train_qa_pairs_kqt_15.json}"
TEST_FILE="${TEST_FILE:-./datasets/${DATASET_NAME}/preprocessed/test_qa_pairs_kqt_15.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/single-lora-target-poi-multiview-${DATASET_NAME}-${VARIANT_NAME}-ablation}"
SEQ_LEN="${SEQ_LEN:-4096}"
THRESHOLD_MIB="${THRESHOLD_MIB:-20000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-./logs/run_single_lora_multiview}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/run_status_${RUN_TS}.log"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
TF32_ENABLED="${TF32_ENABLED:-False}"

mkdir -p "${LOG_DIR}"

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
echo "[$(timestamp)] Dataset name: ${DATASET_NAME}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Variant name: ${VARIANT_NAME}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Train file: ${TRAIN_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Test file: ${TEST_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Output dir: ${OUTPUT_DIR}" | tee -a "${STATUS_LOG}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[$(timestamp)] Missing train file: ${TRAIN_FILE}" | tee -a "${STATUS_LOG}" >&2
  exit 1
fi

if [[ ! -f "${TEST_FILE}" ]]; then
  echo "[$(timestamp)] Missing test file: ${TEST_FILE}" | tee -a "${STATUS_LOG}" >&2
  exit 1
fi

GPU_INFO="$(wait_for_gpu training)"
GPU_ID="$(echo "$GPU_INFO" | cut -d',' -f1)"
GPU_TOTAL="$(echo "$GPU_INFO" | cut -d',' -f2)"
GPU_FREE="$(echo "$GPU_INFO" | cut -d',' -f3)"
GPU_NAME="$(echo "$GPU_INFO" | cut -d',' -f4-)"

echo "[$(timestamp)] Using GPU ${GPU_ID} (${GPU_NAME}) for training (total=${GPU_TOTAL} MiB, free=${GPU_FREE} MiB)" | tee -a "${STATUS_LOG}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

TRAIN_CMD=(
  python supervised-fine-tune-qlora-old.py
  --model_name_or_path "${MODEL_PATH}"
  --bf16 True
  --output_dir "${OUTPUT_DIR}"
  --use_flash_attn False
  --dataset "${TRAIN_FILE}"
  --low_rank_training True
  --num_train_epochs 1
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 2
  --evaluation_strategy no
  --gradient_accumulation_steps 8
  --save_strategy no
  --save_steps 100
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
)

echo "Starting single-LoRA training..."
"${TRAIN_CMD[@]}"
echo "Training finished."

GPU_INFO_AFTER="$(wait_for_gpu evaluation)"
GPU_ID_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f1)"
GPU_TOTAL_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f2)"
GPU_FREE_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f3)"
GPU_NAME_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f4-)"

echo "[$(timestamp)] Starting evaluation on GPU ${GPU_ID_AFTER} (${GPU_NAME_AFTER}) (total=${GPU_TOTAL_AFTER} MiB, free=${GPU_FREE_AFTER} MiB)" | tee -a "${STATUS_LOG}"
export CUDA_VISIBLE_DEVICES="${GPU_ID_AFTER}"

TEST_FILE_BASENAME="$(basename "${TEST_FILE}")"
EVAL_CMD=(
  python eval_next_poi.py
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_name "${DATASET_NAME}"
  --test_file "${TEST_FILE_BASENAME}"
  --context_size "${SEQ_LEN}"
  --seq_len "${SEQ_LEN}"
  --batch_size 1
)

if [[ "${EVAL_DEVICE}" != "auto" ]]; then
  EVAL_CMD+=(--device "${EVAL_DEVICE}")
fi

"${EVAL_CMD[@]}"
echo "Evaluation finished."
