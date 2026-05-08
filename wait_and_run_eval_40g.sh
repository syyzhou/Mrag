#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="./Qwen2.5-3B"
DATASET_NAME="nyc"
OUTPUT_DIR="./output/train_multiview"
TEST_FILE="./rag/multiview_enriched_qa/test_qa_multiview.txt"
THRESHOLD_MIB=40000
POLL_SECONDS=60
LOG_DIR="./logs/wait_eval_40g"

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

RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/wait_status_${RUN_TS}.log"
RUN_LOG="${LOG_DIR}/eval_run_${RUN_TS}.log"
DONE_MARK="${LOG_DIR}/DONE_${RUN_TS}.txt"

echo "[$(timestamp)] Waiting for a GPU with free memory >= ${THRESHOLD_MIB} MiB" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Model path: ${MODEL_PATH}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Output dir: ${OUTPUT_DIR}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Test file: ${TEST_FILE}" | tee -a "${STATUS_LOG}"

while true; do
  GPU_INFO="$(find_gpu_over_threshold || true)"

  if [[ -n "${GPU_INFO}" ]]; then
    GPU_ID="$(echo "${GPU_INFO}" | cut -d',' -f1)"
    GPU_TOTAL="$(echo "${GPU_INFO}" | cut -d',' -f2)"
    GPU_FREE="$(echo "${GPU_INFO}" | cut -d',' -f3)"
    GPU_NAME="$(echo "${GPU_INFO}" | cut -d',' -f4-)"

    echo "[$(timestamp)] Found GPU ${GPU_ID} (${GPU_NAME}) total=${GPU_TOTAL} MiB free=${GPU_FREE} MiB" | tee -a "${STATUS_LOG}"
    echo "[$(timestamp)] Starting evaluation..." | tee -a "${STATUS_LOG}"

    export CUDA_VISIBLE_DEVICES="${GPU_ID}"

    {
      echo "[$(timestamp)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
      echo "[$(timestamp)] Command:"
      echo "python eval_next_poi_loss.py --model_path ${MODEL_PATH} --dataset_name ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --test_file ${TEST_FILE} --device cuda:0"
      python eval_next_poi_loss.py \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --test_file "${TEST_FILE}" \
        --device cuda:0
      EXIT_CODE=$?
      echo "[$(timestamp)] Evaluation finished with exit code ${EXIT_CODE}"
      exit ${EXIT_CODE}
    } 2>&1 | tee -a "${RUN_LOG}"

    exit_code=${PIPESTATUS[0]}
    {
      echo "status=${exit_code}"
      echo "finished_at=$(timestamp)"
      echo "gpu_id=${GPU_ID}"
      echo "gpu_name=${GPU_NAME}"
      echo "run_log=${RUN_LOG}"
      echo "status_log=${STATUS_LOG}"
    } > "${DONE_MARK}"
    echo "[$(timestamp)] Saved run log to ${RUN_LOG}" | tee -a "${STATUS_LOG}"
    echo "[$(timestamp)] Wrote done marker to ${DONE_MARK}" | tee -a "${STATUS_LOG}"
    exit "${exit_code}"
  fi

  echo "[$(timestamp)] No GPU meets threshold yet; sleeping ${POLL_SECONDS}s" | tee -a "${STATUS_LOG}"
  sleep "${POLL_SECONDS}"
done
