# 自动训练和评估脚本使用指南

## 脚本说明

已为您创建两个版本的自动化脚本，可自动检测40GB显存GPU并执行训练和测试：

### 1. **Python版本** (`auto_train_and_eval.py`)
更灵活的Python实现，支持更多自定义参数。

#### 使用方法：

```bash
# 基础使用（使用默认参数）
python auto_train_and_eval.py

# 自定义训练数据和输出目录
python auto_train_and_eval.py \
  --train_data rag/multiview_enriched_qa/train_qa_multiview.json \
  --output_dir ./output_multiview_two_router \
  --test_file test_qa_pairs_kqt_100.txt

# 自定义训练参数
python auto_train_and_eval.py \
  --train_data rag/multiview_enriched_qa/train_qa_multiview.json \
  --output_dir ./output_multiview_two_router \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5

# 仅执行训练（跳过测试）
python auto_train_and_eval.py --skip_eval

# 仅执行测试（跳过训练）
python auto_train_and_eval.py --skip_training \
  --output_dir ./output_multiview_two_router

# 使用轨迹向量进行测试
python auto_train_and_eval.py \
  --trajectory_embedding_path ./precomputed_embeddings.pt
```

#### Python版本参数列表：

```
必需参数:
  --train_data              训练数据路径 (默认: rag/multiview_enriched_qa/train_qa_multiview.json)
  --output_dir              模型输出目录 (默认: ./output_multiview_two_router)
  --test_file               测试文件名 (默认: test_qa_pairs_kqt_100.txt)

训练参数:
  --num_train_epochs        训练轮数 (默认: 3)
  --per_device_train_batch_size  单GPU批大小 (默认: 2)
  --learning_rate           学习率 (默认: 1e-4)
  --gradient_accumulation_steps  梯度累积步数 (默认: 4)

测试参数:
  --trajectory_embedding_path  预编码轨迹向量路径 (可选)

控制参数:
  --skip_training           跳过训练，仅执行测试
  --skip_eval              跳过测试，仅执行训练
```

---

### 2. **Bash版本** (`auto_train_and_eval.sh`)
轻量级的Shell脚本，适合快速执行。

#### 使用方法：

```bash
# 基础使用（使用默认参数）
bash auto_train_and_eval.sh

# 自定义参数（位置参数）
bash auto_train_and_eval.sh \
  "rag/multiview_enriched_qa/train_qa_multiview.json" \
  "./output_multiview_two_router" \
  "test_qa_pairs_kqt_100.txt"
```

#### Bash版本位置参数：
1. **$1** - 训练数据路径 (默认: `rag/multiview_enriched_qa/train_qa_multiview.json`)
2. **$2** - 输出目录 (默认: `./output_multiview_two_router`)
3. **$3** - 测试文件名 (默认: `test_qa_pairs_kqt_100.txt`)

---

## 工作流程

两个脚本的执行流程完全相同：

1. **GPU检测**
   - 自动检测系统中所有GPU的显存大小
   - 找到具有40GB显存的GPU（允许38-42GB范围内的误差）
   - 如果没有找到，脚本会退出并提示错误

2. **训练阶段** (如果未跳过)
   - 使用found的GPU执行Two-Router训练
   - 训练数据: `rag/multiview_enriched_qa/train_qa_multiview.json`
   - 自动保存模型到指定的输出目录
   - 显示训练进度和结果

3. **测试阶段** (如果未跳过)
   - 使用已训练的模型执行评估
   - 计算ACC@1, ACC@5, ACC@10指标
   - 显示测试结果

4. **完成通知**
   - 所有任务完成后显示成功消息

---

## 训练命令详解

两个脚本内部使用的核心训练命令为：

```bash
python -m torch.distributed.launch --nproc_per_node 1 supervised-fine-tune-qlora-two-router.py \
  --model_name_or_path ./Qwen2-1.5B \
  --dataset rag/multiview_enriched_qa/train_qa_multiview.json \
  --output_dir ./output_multiview_two_router \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --warmup_steps 100 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --save_total_limit 3 \
  --seed 42 \
  --model_max_length 8192 \
  --use_flash_attn True \
  --low_rank_training True \
  --lora_r 8 \
  --lora_alpha 16 \
  --num_experts 4 \
  --use_hydra_lora True \
  --use_trajectory_routing False \
  --ddp_find_unused_parameters False
```

主要特点：
- 使用分布式训练框架 (单GPU)
- 40GB显存优化的配置
- LoRA适配器微调
- 4个专家的多任务路由
- 不使用轨迹路由（可通过参数启用）

---

## 测试命令详解

测试命令为：

```bash
python eval_next_poi_loss.py \
  --batch_size 1 \
  --base_model ./Qwen2-1.5B \
  --seq_len 8192 \
  --context_size 8192 \
  --peft_model ./output_multiview_two_router \
  --model_path ./output_multiview_two_router \
  --output_dir ./output_multiview_two_router \
  --test_file test_qa_pairs_kqt_100.txt \
  --dataset_name test
```

---

## 高级用法示例

### 示例1：完整的多参数训练

```bash
python auto_train_and_eval.py \
  --train_data rag/multiview_enriched_qa/train_qa_multiview.json \
  --output_dir ./output_multiview_two_router_v2 \
  --test_file test_qa_pairs_kqt_100.txt \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps 2
```

### 示例2：使用轨迹向量的测试

```bash
python auto_train_and_eval.py \
  --skip_training \
  --output_dir ./output_multiview_two_router \
  --trajectory_embedding_path ./precomputed_trajectory_embeddings.pt \
  --test_file test_qa_pairs_with_trajectories.txt
```

### 示例3：快速验证（训练一个epoch）

```bash
python auto_train_and_eval.py \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1
```

---

## 输出示例

脚本执行时会显示：

```
================================================================================
自动训练和评估脚本
时间: 2026-04-16 20:13:45
================================================================================

正在检测GPU显存...
✓ 检测到 1 个GPU
  GPU 0: NVIDIA H100 PCIe - 40.0GB
✓ GPU 0 有约40GB显存 (40.0GB) - 将使用此GPU进行训练

================================================================================
执行Two-Router训练
================================================================================

[训练输出...]

✓ 训练成功完成

================================================================================
执行测试
================================================================================

[测试输出...]

Results:
  Total samples: 100, Skipped: 0
  ACC@1: 0.6200 (62/100)
  ACC@5: 0.8900 (89/100)
  ACC@10: 0.9500 (95/100)

✓ 测试成功完成

================================================================================
✓ 所有任务完成
================================================================================
```

---

## 常见问题

### Q: 脚本找不到40GB显卡怎么办？
A: 检查以下几点：
1. 确保CUDA已正确安装：`nvidia-smi`
2. 检查显卡显存大小：`nvidia-smi --query-gpu=memory.total --format=csv`
3. 允许的显存范围是38-42GB，如果您的显卡不在此范围，需要修改脚本

### Q: 训练过程中出错怎么办？
A: 
1. 检查训练数据文件是否存在
2. 查看输出目录是否有写权限
3. 检查显存是否足够（40GB应该足够）
4. 可以使用 `--skip_eval` 跳过测试，只运行训练

### Q: 如何修改训练参数？
A:
- **Python版本**：使用命令行参数如 `--num_train_epochs 5`
- **Bash版本**：直接修改脚本中的硬编码参数

### Q: 可以同时训练多个模型吗？
A: 可以，但需要确保显存足够。建议使用不同的输出目录来区分模型。

---

## 文件结构

```
LLM4POI/
├── auto_train_and_eval.py          # Python版本脚本
├── auto_train_and_eval.sh          # Bash版本脚本
├── supervised-fine-tune-qlora-two-router.py  # 训练脚本
├── eval_next_poi_loss.py           # 测试脚本
├── rag/
│   └── multiview_enriched_qa/
│       └── train_qa_multiview.json  # 训练数据
└── output_multiview_two_router/     # 输出目录（自动创建）
    ├── adapter_model.safetensors
    ├── training_args.bin
    └── ...
```

---

## 推荐用法

对于您的场景（40GB显卡，使用multiview数据，执行two-router训练后测试），推荐使用：

```bash
# 最简单的方法
python auto_train_and_eval.py

# 或者使用Bash版本
bash auto_train_and_eval.sh
```

这两个命令都会自动：
1. ✓ 检测40GB GPU
2. ✓ 执行Two-Router训练（数据源：`rag/multiview_enriched_qa/train_qa_multiview.json`）
3. ✓ 自动执行测试
4. ✓ 显示最终结果

---

## 关键特性

✓ **自动GPU检测** - 无需手动指定GPU设备
✓ **一键执行** - 从训练到测试自动流程
✓ **可自定义** - 支持灵活的参数配置
✓ **错误处理** - 异常时自动退出并报告
✓ **进度显示** - 实时显示执行进度
✓ **灵活跳过** - 支持仅训练或仅测试

---

创建时间: 2026-04-16
