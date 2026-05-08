#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-nyc}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
DROPOUT="${DROPOUT:-0.4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-3}"
LR="${LR:-3e-4}"
RANK_WEIGHT="${RANK_WEIGHT:-0.0}"
ALIGN_WEIGHT="${ALIGN_WEIGHT:-0.0}"
DEVICE="${DEVICE:-cuda}"
FUSION_TYPE="${FUSION_TYPE:-cross_attn}"
GRAPH_ALIGN_RESIDUAL="${GRAPH_ALIGN_RESIDUAL:-True}"
SAVE_DIR="${SAVE_DIR:-./rag/target_poi_multiview/artifacts_graph_align_cross_attn_do04_wd3e3_${DATASET_NAME}}"

python rag/target_poi_multiview/train.py \
  --dataset_name "${DATASET_NAME}" \
  --device "${DEVICE}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --dropout "${DROPOUT}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --fusion_type "${FUSION_TYPE}" \
  --fusion_heads 4 \
  --graph_alignment context_cross_attn \
  --graph_align_heads 4 \
  --graph_align_residual "${GRAPH_ALIGN_RESIDUAL}" \
  --rank_weight "${RANK_WEIGHT}" \
  --align_weight "${ALIGN_WEIGHT}" \
  --save_dir "${SAVE_DIR}"
