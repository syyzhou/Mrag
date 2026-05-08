# LLM4POI Methodology Notes

本文档整理 `run_two_router_multiview_random10_global10.sh` 对应实验链路，作为后续论文方法论写作的项目备忘。

## 1. 总体流程

该实验链路由三部分组成：

1. 轨迹数据构建为 next-POI QA 样本。
2. 训练并使用 target-POI multi-view retriever，为 QA prompt 追加候选 POI。
3. 使用带候选 POI 的 QA 数据对 two-router LLM 进行监督微调，并在测试集上生成下一个 POI id。

`run_two_router_multiview_random10_global10.sh` 是最终微调和评测入口。它本身不重新构建 QA 数据、不重新训练检索器、不重新导出候选集，也不重新预编码轨迹向量，而是使用已经生成好的数据文件和 embedding 文件。

## 2. QA 数据构建

原始轨迹数据来自 `datasets/nyc/preprocessed/train_sample.csv` 和 `datasets/nyc/preprocessed/test_sample.csv`/`test_sample_with_traj.csv`。预处理逻辑主要在 `preprocessing/to_nextpoi_kqt.py` 中。

每条样本以一条用户轨迹为单位构造 next-POI prediction 任务：

- 当前轨迹中除最后一个 check-in 外的访问序列作为输入。
- 当前轨迹最后一个 POI 作为预测目标。
- prompt 包含用户 id、当前轨迹、历史轨迹、目标时间，以及 POI id 的合法范围。
- answer 格式为目标时间、用户 id、真实 POI id 和 POI 类别。

测试集不是只使用测试自身历史。当前逻辑会把同一用户的训练历史轨迹作为已知历史拼接进测试样本的历史上下文，再结合测试集中当前轨迹之前的历史轨迹构造 prompt。

最终基础 QA 文件规模：

- `datasets/nyc/preprocessed/train_qa_pairs_kqt.json`: 11022 条训练样本。
- `datasets/nyc/preprocessed/test_qa_pairs_kqt.json`: 1447 条测试样本。

## 3. 多视图 Target-POI 检索器

检索器代码位于 `rag/target_poi_multiview/train.py`。它是一个直接面向目标 POI 的 multi-view retriever，用于给 LLM prompt 提供候选 POI。

### 3.1 POI 表示

每个 POI embedding 由三类信息融合得到：

- trainable POI id embedding。
- 缓存的语义向量，来自 `rag/feature_cache/bert_${DATASET_NAME}/poi_sem_vectors.npy`。
- 地理特征，来自 `rag/feature_cache/bert_${DATASET_NAME}/poi_geo_features.npy`。

feature cache 元信息记录在 `rag/feature_cache/bert_${DATASET_NAME}/feature_meta.json`：

- 编码器：BERT。
- POI 数量：4980。
- 语义向量维度：768。
- 地理特征维度：6。
- POI 顺序：`sorted(dataset.poi_dict.keys())`。

### 3.2 Query 多视图编码

检索器包含三个 query view：

- Semantic view：使用最后访问 POI embedding、目标小时、星期、时间桶编码短期语义意图。
- Trajectory view：使用 Transformer encoder 编码当前轨迹 POI 序列及时间特征。
- Structure view：基于时段 transition graph，在最后访问 POI 周围构造局部子图，并用 GAT 编码结构转移模式。

三个 view 输出后经过 gated fusion，得到 fused query embedding。

### 3.3 检索器训练目标

训练时使用所有 POI embedding 作为候选库，令 query 检索真实 target POI。损失包括：

- fused query 到真实 target POI 的主交叉熵损失。
- trajectory/semantic/structure 三个单视图 query 的辅助交叉熵损失。
- 可选的 view alignment loss，用于约束不同 view 的表示一致性。

当前 random10_global10 实验对应的检索器 artifact 为：

- `rag/target_poi_multiview/artifacts_dropout04_wd3e3/model.pth`

文件名和实验记录表明该版本使用：

- dropout = 0.4
- weight_decay = 3e-3

## 4. 候选 POI 导出与 Prompt 增强

候选导出逻辑位于 `rag/target_poi_multiview/export_candidates.py`。

导出时，检索器对每条 QA 样本输出候选 POI，并将候选追加到 question 末尾。追加文本会说明这些候选只是 supplementary references，真实下一个 POI 不一定包含在候选中。

当前实验文件：

- 训练文件：`datasets/nyc/preprocessed/train_qa_pairs_kqt_candidates_target_poi_multiview_dropout04_wd3e3_train_random10_global10_test_top20_uncertain.json`
- 测试文件：`datasets/nyc/preprocessed/test_qa_pairs_kqt_candidates_target_poi_multiview_dropout04_wd3e3_train_random10_global10_test_top20_uncertain.txt`

候选规则记录在：

- `datasets/nyc/preprocessed/target_poi_multiview_dropout04_wd3e3_train_random10_global10_test_top20_uncertain_candidate_stats.json`

具体规则：

- `top_k = 20`
- `retrieval_top_k = 20`
- 训练集：从检索 top20 中随机保留 10 个候选，再补充 10 个训练答案中的全局热门 POI。
- 测试集：使用自然检索 top20，不补充 global popular candidates。
- 不强制插入真实 target POI。

统计结果：

- 训练集样本数：11022。
- 测试集样本数：1447。
- 训练集 target 出现在候选中的比例：0.5244。
- 测试集 target 出现在候选中的比例：0.4126。
- 测试集有 18 条样本未成功构建候选，因此存在候选数量不足的记录。

## 5. 轨迹向量预编码

two-router LLM 微调和评测会使用外部预编码轨迹向量：

- 训练轨迹 embedding：`datasets/nyc/preprocessed/trajectory_embeddings.pt`
- 测试轨迹 embedding：`datasets/nyc/preprocessed/test_embeddings.pt`

相关脚本为 `precompute_trajectory_embeddings.py`。

预编码流程：

- 从 QA question 中提取轨迹相关文本。
- 使用基础 LLM 编码文本。
- 取最后一层 hidden states。
- 根据 pooling 策略得到每条样本的轨迹级 embedding。
- 保存 embedding、模型名、维度、样本数和提取匹配统计。

这些 embedding 不直接拼接进 prompt，而是在 two-router 模型内部作为 trajectory routing 信号使用。

## 6. Two-Router LLM 微调

最终训练入口为 `run_two_router_multiview_random10_global10.sh`，实际调用：

- `supervised-fine-tune-qlora-two-router.py`

基础模型：

- `./Qwen2.5-3B`

输出目录：

- `output/two-router-target-poi-multiview-dropout04-wd3e3-train-random10-global10-test-top20-uncertain`

主要训练参数：

- bf16 = True
- num_train_epochs = 1
- per_device_train_batch_size = 1
- gradient_accumulation_steps = 8
- learning_rate = 2e-5
- warmup_steps = 20
- lr_scheduler_type = constant_with_warmup
- model_max_length = 4096
- DeepSpeed config = `ds_configs/stage2.json`
- gradient_checkpointing = True
- save_strategy = no

### 6.1 Two-router / HMORA 机制

模型改造逻辑主要在 `model8.py` 和 `config.py`。

基础 LLM 参数被冻结，只训练 HMORA/LoRA 相关可训练参数。每个目标线性层被替换为带多专家 LoRA 的模块。每层有两组 router 和一个组间 fusion gate：

- Router 1：基于 token hidden states 的平均池化表示进行路由。
- Router 2：使用外部 trajectory embedding 经 `TrajectoryProjector` 投影后进行路由。

两组 router 分别选择/加权本组 LoRA experts，专家输出再通过 fusion gate 融合。当前脚本没有启用 QKV 或 FFN 内部的 router 共享；`share_traj_projector = False` 也表示 trajectory projector 不跨层共享。

另外，Router 1 启用了 shared expert。也就是说，Router 1 除了根据 token hidden states 动态路由到多个专家外，还额外叠加一个共享专家输出，用于保留所有样本都可使用的公共适配能力。共享专家权重由 `router1_shared_expert_weight` 控制，当前为 1.0。

当前脚本启用：

- `use_trajectory_routing = True`
- `trajectory_fusion_mode = gate`
- `share_traj_projector = False`
- `router1_use_shared_expert = True`
- `router1_shared_expert_weight = 1.0`

训练时，`SupervisedDatasetWithEmbeddings` 会为每条训练样本加载对应 trajectory embedding。`UmraTrainer.compute_loss` 在 forward 前把当前 batch 的 trajectory embedding 写入所有 trajectory-enabled router 的 projector 缓存。forward 后清理 routing cache。

## 7. 测试与评测

脚本训练结束后调用：

- `eval_two_router.py`

评测输入：

- base model: `./Qwen2.5-3B`
- adapter/output_dir: `output/two-router-target-poi-multiview-dropout04-wd3e3-train-random10-global10-test-top20-uncertain`
- test file: `datasets/nyc/preprocessed/test_qa_pairs_kqt_candidates_target_poi_multiview_dropout04_wd3e3_train_random10_global10_test_top20_uncertain.txt`
- test trajectory embedding: `datasets/nyc/preprocessed/test_embeddings.pt`

评测流程：

1. 加载基础模型。
2. 从 output dir 加载 two-router/HMORA adapter。
3. 对每条测试样本，将对应 test trajectory embedding 写入 trajectory router。
4. 构造生成 prompt，格式截止到 `will visit POI id`。
5. 使用 beam search 生成 POI id。
6. 从生成结果中解析数字 POI id，并和 ground truth 比较。

生成配置：

- max_new_tokens = 30
- do_sample = False
- num_beams = 5
- num_return_sequences = 5
- repetition_penalty = 1.176

评测指标：

- ACC@1
- ACC@5
- ACC@10

当前 `eval_two_router.py` 中 ACC 分母使用测试总样本数，而不是去掉过长跳过样本后的 evaluated 数量。

## 8. 可用于论文方法论的抽象表述

该方法可以概括为：

1. 将用户移动轨迹转化为带历史上下文的 next-POI instruction tuning 样本。
2. 构建一个目标 POI 级多视图检索器，从语义、轨迹序列和结构转移三个角度召回候选 POI。
3. 将检索候选作为软参考注入 LLM prompt，而不假设真实 POI 必然在候选集中。
4. 使用 two-router 参数高效微调框架，使 LLM 同时利用 token-level textual context 和外部 trajectory-level embedding。
5. 通过 trajectory-aware routing 动态调节 LoRA experts，使不同移动模式的样本激活不同专家组合。
6. 最终由 LLM 在候选增强 prompt 和 trajectory routing 信号共同作用下生成下一个 POI id。
