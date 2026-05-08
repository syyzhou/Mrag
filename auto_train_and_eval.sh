#!/bin/bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TRAIN_DATA="${1:-rag/multiview_enriched_qa_new/train_qa_multiview.json}"
OUTPUT_DIR="${2:-./output/train_multiview_new}"
TEST_FILE="${3:-rag/multiview_enriched_qa_new/test_qa_multiview.txt}"
TRAIN_SCRIPT="${4:-supervised-fine-tune-qlora-two-router.py}"
MODEL_NAME_OR_PATH="${5:-./Qwen2.5-3B}"

THRESHOLD_MIB="${THRESHOLD_MIB:-20000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-./logs/auto_train_and_eval}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/status_${RUN_TS}.log"

mkdir -p "${LOG_DIR}"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

cat <<EOF
${BLUE}======================================================================${NC}
${BLUE}自动训练和评估脚本${NC}
${BLUE}======================================================================${NC}
训练数据: ${TRAIN_DATA}
输出目录: ${OUTPUT_DIR}
测试文件: ${TEST_FILE}
训练脚本: ${TRAIN_SCRIPT}
模型路径: ${MODEL_NAME_OR_PATH}
显存阈值: ${THRESHOLD_MIB} MiB
轮询间隔: ${POLL_SECONDS}s
EOF

echo ""

find_gpu_over_threshold() {
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
        gpu_info="$(find_gpu_over_threshold || true)"
        if [[ -n "${gpu_info}" ]]; then
            echo "${gpu_info}"
            return 0
        fi
        echo "[$(timestamp)] No suitable GPU for ${stage}; sleeping ${POLL_SECONDS}s" | tee -a "${STATUS_LOG}" >&2
        sleep "${POLL_SECONDS}"
    done
}

echo -e "${YELLOW}正在检测可用GPU显存...${NC}"
GPU_INFO="$(wait_for_gpu training)"
GPU_ID="$(echo "${GPU_INFO}" | cut -d',' -f1)"
GPU_TOTAL="$(echo "${GPU_INFO}" | cut -d',' -f2)"
GPU_FREE="$(echo "${GPU_INFO}" | cut -d',' -f3)"
GPU_NAME="$(echo "${GPU_INFO}" | cut -d',' -f4-)"

echo -e "${GREEN}✓ 检测到GPU ${GPU_ID} (${GPU_NAME})，总显存 ${GPU_TOTAL} MiB，空闲显存 ${GPU_FREE} MiB${NC}"
echo ""

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}执行Two-Router训练${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

TRAIN_CMD="python $TRAIN_SCRIPT \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --bf16=True \
  --output_dir $OUTPUT_DIR \
  --use_flash_attn=False \
  --dataset $TRAIN_DATA \
  --low_rank_training True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --evaluation_strategy no \
  --gradient_accumulation_steps 8 \
  --save_strategy no \
  --save_steps 100 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --lr_scheduler_type constant_with_warmup \
  --logging_steps 1 \
  --logging_dir ./logs \
  --deepspeed ds_configs/stage2.json \
  --model_max_length 4096 \
  --tf32 True \
  --use_trajectory_routing True \
  --trajectory_embedding_path ./datasets/nyc/preprocessed/trajectory_embeddings.pt \
  --trajectory_fusion_mode gate \
  --share_traj_projector False \
  --router1_use_shared_expert True \
  --router1_shared_expert_weight 1.0 \
  --dataset_name nyc \
  --dropout 0.05"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if eval "$TRAIN_CMD"; then
    echo -e "${GREEN}✓ 训练成功完成${NC}"
else
    echo -e "${RED}❌ 训练失败${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}正在等待可用GPU执行测试...${NC}"
GPU_INFO="$(wait_for_gpu evaluation)"
GPU_ID="$(echo "${GPU_INFO}" | cut -d',' -f1)"
GPU_TOTAL="$(echo "${GPU_INFO}" | cut -d',' -f2)"
GPU_FREE="$(echo "${GPU_INFO}" | cut -d',' -f3)"
GPU_NAME="$(echo "${GPU_INFO}" | cut -d',' -f4-)"

echo -e "${GREEN}✓ 测试将使用GPU ${GPU_ID} (${GPU_NAME})，总显存 ${GPU_TOTAL} MiB，空闲显存 ${GPU_FREE} MiB${NC}"
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}执行测试${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

EVAL_CMD="python eval_two_router.py \
  --batch_size 1 \
  --base_model ./Qwen2.5-3B \
  --seq_len 4096 \
  --context_size 4096 \
  --model_path ./Qwen2.5-3B \
  --output_dir $OUTPUT_DIR \
  --test_file $TEST_FILE \
  --dataset_name nyc \
  --trajectory_embedding_path ./datasets/nyc/preprocessed/test_embeddings.pt"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if eval "$EVAL_CMD"; then
    echo -e "${GREEN}✓ 测试成功完成${NC}"
else
    echo -e "${RED}❌ 测试失败${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}✓ 所有任务完成${NC}"
echo -e "${BLUE}======================================================================${NC}"
