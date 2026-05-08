#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="./Qwen2.5-3B"
SEQ_LEN=4096
DATASET_NAME="${DATASET_NAME:-nyc}"
CANDIDATE_TAG="${CANDIDATE_TAG:-dst_time_layered12_far3_mismatch3_tail2_test}"
TRAIN_FILE="./datasets/${DATASET_NAME}/preprocessed/train_qa_pairs_kqt_candidates_target_poi_multiview_${CANDIDATE_TAG}.json"
TEST_FILE="./datasets/${DATASET_NAME}/preprocessed/test_qa_pairs_kqt_candidates_target_poi_multiview_${CANDIDATE_TAG}.txt"
OUTPUT_DIR="./output/two-router-target-poi-multiview-${DATASET_NAME}-${CANDIDATE_TAG}"
TRAIN_THRESHOLD_MIB="${TRAIN_THRESHOLD_MIB:-20000}"
EVAL_THRESHOLD_MIB="${EVAL_THRESHOLD_MIB:-28000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-./logs/run_two_router_multiview_layered}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/run_status_${RUN_TS}.log"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
TF32_ENABLED="${TF32_ENABLED:-False}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-30}"
EVAL_NUM_BEAMS="${EVAL_NUM_BEAMS:-5}"
EVAL_NUM_RETURN_SEQUENCES="${EVAL_NUM_RETURN_SEQUENCES:-5}"
EVAL_REPETITION_PENALTY="${EVAL_REPETITION_PENALTY:-1.176}"
EVAL_LENGTH_PENALTY="${EVAL_LENGTH_PENALTY:-1.0}"
RETRIEVER_ARTIFACT_TAG="${RETRIEVER_ARTIFACT_TAG:-dropout04_wd3e3_${DATASET_NAME}}"
RETRIEVER_ARTIFACT_DIR="./rag/target_poi_multiview/artifacts_${RETRIEVER_ARTIFACT_TAG}"
FEATURE_CACHE_NAME="${FEATURE_CACHE_NAME:-bert_${DATASET_NAME}}"
FEATURE_CACHE_DIR="./rag/feature_cache/${FEATURE_CACHE_NAME}"
FORCE_REBUILD_CANDIDATES="${FORCE_REBUILD_CANDIDATES:-0}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-${MODEL_PATH}}"

# Common embedding paths.
TRAIN_TRAJ_EMB="./datasets/${DATASET_NAME}/preprocessed/trajectory_embeddings.pt"
TEST_TRAJ_EMB="./datasets/${DATASET_NAME}/preprocessed/test_embeddings.pt"

mkdir -p "${LOG_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

find_gpu_over_threshold() {
  local threshold_mib="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$(timestamp)] nvidia-smi not found; cannot detect GPU availability." | tee -a "${STATUS_LOG}" >&2
    return 1
  fi

  nvidia-smi --query-gpu=index,memory.total,memory.free,name --format=csv,noheader,nounits \
  | awk -F', ' -v threshold="${threshold_mib}" '$3 >= threshold {print $1","$2","$3","$4}' \
  | sort -t',' -k3,3nr \
  | head -n 1
}

wait_for_gpu() {
  local stage="$1"
  local threshold_mib="$2"
  echo "[$(timestamp)] Waiting for GPU for ${stage} with free memory >= ${threshold_mib} MiB" | tee -a "${STATUS_LOG}" >&2
  while true; do
    local gpu_info
    if ! gpu_info="$(find_gpu_over_threshold "${threshold_mib}" 2>/dev/null)"; then
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
echo "[$(timestamp)] Retriever artifact dir: ${RETRIEVER_ARTIFACT_DIR}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Feature cache dir: ${FEATURE_CACHE_DIR}" | tee -a "${STATUS_LOG}"

ensure_parent_dir() {
  local fpath="$1"
  mkdir -p "$(dirname "${fpath}")"
}

run_retriever_pipeline_if_needed() {
  if [[ ! -d "${FEATURE_CACHE_DIR}" ]]; then
    echo "[$(timestamp)] ERROR: feature cache dir not found: ${FEATURE_CACHE_DIR}" | tee -a "${STATUS_LOG}" >&2
    echo "[$(timestamp)] Please build cache first, e.g.:" | tee -a "${STATUS_LOG}" >&2
    echo "[$(timestamp)] python rag/build_poi_feature_cache.py --train_csv ./datasets/${DATASET_NAME}/preprocessed/train_sample.csv --test_csv ./datasets/${DATASET_NAME}/preprocessed/test_sample_with_traj.csv --train_qa ./datasets/${DATASET_NAME}/preprocessed/train_qa_pairs_kqt.json --test_qa ./datasets/${DATASET_NAME}/preprocessed/test_qa_pairs_kqt.json --output_dir ${FEATURE_CACHE_DIR} --encoder bert --device cuda" | tee -a "${STATUS_LOG}" >&2
    return 1
  fi

  if [[ "${FORCE_REBUILD_CANDIDATES}" == "1" ]]; then
    echo "[$(timestamp)] FORCE_REBUILD_CANDIDATES=1, will rebuild retriever and candidate files." | tee -a "${STATUS_LOG}"
  elif [[ -f "${TRAIN_FILE}" && -f "${TEST_FILE}" ]]; then
    echo "[$(timestamp)] Candidate files already exist; skip retriever training/export." | tee -a "${STATUS_LOG}"
    return 0
  fi

  ensure_parent_dir "${TRAIN_FILE}"
  ensure_parent_dir "${TEST_FILE}"
  ensure_parent_dir "${RETRIEVER_ARTIFACT_DIR}/model.pth"

  echo "[$(timestamp)] Candidate files missing; start retriever training." | tee -a "${STATUS_LOG}"
  python rag/target_poi_multiview/train.py \
    --dataset_name "${DATASET_NAME}" \
    --feature_cache "${FEATURE_CACHE_DIR}" \
    --save_dir "${RETRIEVER_ARTIFACT_DIR}" \
    --dropout 0.4 \
    --weight_decay 3e-3

  echo "[$(timestamp)] Retriever training finished; exporting layered candidates." | tee -a "${STATUS_LOG}"
  python rag/target_poi_multiview/export_layered_candidates.py \
    --dataset_name "${DATASET_NAME}" \
    --candidate_tag "${CANDIDATE_TAG}" \
    --feature_cache "${FEATURE_CACHE_DIR}" \
    --artifact_dir "${RETRIEVER_ARTIFACT_DIR}" \
    --train_output "${TRAIN_FILE}" \
    --test_output "${TEST_FILE}" \
    --stats_output "./datasets/${DATASET_NAME}/preprocessed/target_poi_multiview_${CANDIDATE_TAG}_stats.json" \
    --test_use_top12

  if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" ]]; then
    echo "[$(timestamp)] ERROR: candidate export failed. train/test file not found after export." | tee -a "${STATUS_LOG}" >&2
    return 1
  fi
  echo "[$(timestamp)] Candidate export finished: ${TRAIN_FILE} and ${TEST_FILE}" | tee -a "${STATUS_LOG}"
}

ensure_trajectory_embeddings() {
  local emb_gpu_info emb_gpu_id emb_gpu_total emb_gpu_free emb_gpu_name

  if [[ ! -f "${TRAIN_TRAJ_EMB}" ]]; then
    emb_gpu_info="$(wait_for_gpu embedding_train "${TRAIN_THRESHOLD_MIB}")"
    emb_gpu_id="$(echo "$emb_gpu_info" | cut -d',' -f1)"
    emb_gpu_total="$(echo "$emb_gpu_info" | cut -d',' -f2)"
    emb_gpu_free="$(echo "$emb_gpu_info" | cut -d',' -f3)"
    emb_gpu_name="$(echo "$emb_gpu_info" | cut -d',' -f4-)"
    echo "[$(timestamp)] Using GPU ${emb_gpu_id} (${emb_gpu_name}) for train embedding precompute (total=${emb_gpu_total} MiB, free=${emb_gpu_free} MiB)" | tee -a "${STATUS_LOG}"
    export CUDA_VISIBLE_DEVICES="${emb_gpu_id}"
    echo "[$(timestamp)] Missing train trajectory embeddings: ${TRAIN_TRAJ_EMB}. Start precompute." | tee -a "${STATUS_LOG}"
    python precompute_trajectory_embeddings.py \
      --dataset_name "${DATASET_NAME}" \
      --split train \
      --model_name_or_path "${EMBED_MODEL_PATH}"
  fi

  if [[ ! -f "${TEST_TRAJ_EMB}" ]]; then
    emb_gpu_info="$(wait_for_gpu embedding_test "${TRAIN_THRESHOLD_MIB}")"
    emb_gpu_id="$(echo "$emb_gpu_info" | cut -d',' -f1)"
    emb_gpu_total="$(echo "$emb_gpu_info" | cut -d',' -f2)"
    emb_gpu_free="$(echo "$emb_gpu_info" | cut -d',' -f3)"
    emb_gpu_name="$(echo "$emb_gpu_info" | cut -d',' -f4-)"
    echo "[$(timestamp)] Using GPU ${emb_gpu_id} (${emb_gpu_name}) for test embedding precompute (total=${emb_gpu_total} MiB, free=${emb_gpu_free} MiB)" | tee -a "${STATUS_LOG}"
    export CUDA_VISIBLE_DEVICES="${emb_gpu_id}"
    echo "[$(timestamp)] Missing test trajectory embeddings: ${TEST_TRAJ_EMB}. Start precompute." | tee -a "${STATUS_LOG}"
    python precompute_trajectory_embeddings.py \
      --dataset_name "${DATASET_NAME}" \
      --split test \
      --model_name_or_path "${EMBED_MODEL_PATH}"
  fi

  if [[ ! -f "${TRAIN_TRAJ_EMB}" || ! -f "${TEST_TRAJ_EMB}" ]]; then
    echo "[$(timestamp)] ERROR: trajectory embedding precompute failed." | tee -a "${STATUS_LOG}" >&2
    return 1
  fi
  echo "[$(timestamp)] Trajectory embeddings ready." | tee -a "${STATUS_LOG}"
}

run_retriever_pipeline_if_needed
ensure_trajectory_embeddings

GPU_INFO="$(wait_for_gpu training "${TRAIN_THRESHOLD_MIB}")"
GPU_ID="$(echo "$GPU_INFO" | cut -d',' -f1)"
GPU_TOTAL="$(echo "$GPU_INFO" | cut -d',' -f2)"
GPU_FREE="$(echo "$GPU_INFO" | cut -d',' -f3)"
GPU_NAME="$(echo "$GPU_INFO" | cut -d',' -f4-)"

echo "[$(timestamp)] Using GPU ${GPU_ID} (${GPU_NAME}) for training (total=${GPU_TOTAL} MiB, free=${GPU_FREE} MiB)" | tee -a "${STATUS_LOG}"
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
  --use_trajectory_routing True
  --trajectory_fusion_mode gate
  --share_traj_projector False
  --router1_use_shared_expert True
  --router1_shared_expert_weight 1.0
)

if [[ -n "${TRAIN_TRAJ_EMB}" ]]; then
  TRAIN_CMD+=(--trajectory_embedding_path "${TRAIN_TRAJ_EMB}")
fi

echo "[$(timestamp)] Starting LLM training." | tee -a "${STATUS_LOG}"
"${TRAIN_CMD[@]}"
echo "[$(timestamp)] LLM training finished." | tee -a "${STATUS_LOG}"

GPU_INFO_EVAL="$(wait_for_gpu evaluation "${EVAL_THRESHOLD_MIB}")"
GPU_ID_EVAL="$(echo "$GPU_INFO_EVAL" | cut -d',' -f1)"
GPU_TOTAL_EVAL="$(echo "$GPU_INFO_EVAL" | cut -d',' -f2)"
GPU_FREE_EVAL="$(echo "$GPU_INFO_EVAL" | cut -d',' -f3)"
GPU_NAME_EVAL="$(echo "$GPU_INFO_EVAL" | cut -d',' -f4-)"
echo "[$(timestamp)] Starting evaluation on GPU ${GPU_ID_EVAL} (${GPU_NAME_EVAL}) (total=${GPU_TOTAL_EVAL} MiB, free=${GPU_FREE_EVAL} MiB)" | tee -a "${STATUS_LOG}"
export CUDA_VISIBLE_DEVICES="${GPU_ID_EVAL}"

EVAL_CMD=(
  python eval_two_router.py
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --test_file "${TEST_FILE}"
  --dataset_name "${DATASET_NAME}"
  --context_size "${SEQ_LEN}"
  --seq_len "${SEQ_LEN}"
  --batch_size 1
  --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
  --num_beams "${EVAL_NUM_BEAMS}"
  --num_return_sequences "${EVAL_NUM_RETURN_SEQUENCES}"
  --repetition_penalty "${EVAL_REPETITION_PENALTY}"
  --length_penalty "${EVAL_LENGTH_PENALTY}"
)

if [[ "${EVAL_DEVICE}" != "auto" ]]; then
  EVAL_CMD+=(--device "${EVAL_DEVICE}")
fi

if [[ -n "${TEST_TRAJ_EMB}" ]]; then
  EVAL_CMD+=(--trajectory_embedding_path "${TEST_TRAJ_EMB}")
fi

"${EVAL_CMD[@]}"
echo "Evaluation finished."
