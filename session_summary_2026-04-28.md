# LLM4POI Session Summary (2026-04-28)

## 项目目标
最终目标不是单纯提高检索模块指标，而是通过检索模块给 Qwen2.5-3B two-router 微调数据拼接候选 POI，从而提高最终 LLM 的 `Acc@1 / Acc@5`。

## 总体流程
1. 原始 train/test QA：
   - `datasets/nyc/preprocessed/train_qa_pairs_kqt.json`
   - `datasets/nyc/preprocessed/test_qa_pairs_kqt.json` / txt
2. 训练 target-POI multiview retriever
3. 用 retriever 给每条 QA 拼接 Candidate POIs
4. 用候选增强后的 train JSON 微调 Qwen2.5-3B two-router
5. 用候选增强后的 test TXT 做 `eval_two_router.py`

## 检索模块与核心文件
- 目录：`rag/target_poi_multiview/`
- 训练主文件：`rag/target_poi_multiview/train.py`
- 候选导出：
  - `rag/target_poi_multiview/export_candidates.py`
  - `rag/target_poi_multiview/export_mixed_candidates.py`
  - `rag/target_poi_multiview/export_layered_candidates.py`

## 关键修复（时间泄露相关）
原来转移索引用的是：
```python
idx.add(src_v.epoch, bucket(src_v), src, dst)
```
已改为：
```python
idx.add(dst_v.epoch, bucket(dst_v), src, dst)
```
理由：`src->dst` 转移被“观察到”的时间应是到达 `dst` 的时间，避免当前样本答案边因 `src_v.epoch < target_epoch` 提前可见。

## 当前主用 retriever 权重
- `rag/target_poi_multiview/artifacts_dropout04_wd3e3_dst_time/model.pth`
- 200 轮训练，best epoch 123，best test `Recall@20 = 0.4213`

## 随机候选基线（最早实验）
代码：
- `datasets/nyc/preprocessed/build_random_candidate_retrieval.py`

核心逻辑：
- 每条样本随机 20 候选
- 用 `include_ratio` 控制 target 在候选中的占比

文件：
- train 60%：`train_qa_pairs_kqt_candidates_enriched.json`
- test 60%：`test_qa_pairs_kqt_candidates_enriched.txt`
- test 30%：`test_qa_pairs_kqt_candidates_enriched_30pct.txt`
- test 30% + uncertain 提示：`test_qa_pairs_kqt_candidates_enriched_30pct_uncertain.txt`

## 分布与现象结论
随机 30% 测试结果有时更好，主要因为候选更“容易排除”：
- 随机候选里大量明显错误项（弱负样本）
- 真实 retriever top20 则更相似（近距离、同类更多），LLM 更难精排

## 候选构造实验要点
### `12 + 3 + 3 + 2` 混合版
- 12：从 retriever top20 随机抽 12 strong
- 3：同类别远距离负样本
- 3：类别不匹配负样本
- 2：retriever rank>=101 的 tail 低分候选

训练统计（已生成版本）：
- train target-in-candidates ≈ 0.5967

## prompt 设计当前结论
为了减少 train/test 分布偏移：
- 不建议强依赖 `source` 标签
- 倾向统一保留：
  - `retrieval_rank`
  - `confidence`
  - `distance`
  - `category`

建议统一候选格式：
```text
POI id(category, distance, retrieval_rank, confidence)
```
规则：
- 检索候选：`rank=1..N`, `confidence` 用样本内 top20 min-max 归一化
- 非检索补充候选：`rank=NA`, `confidence=0.00`

## confidence 计算建议
不直接用原始 score，使用样本内归一化：
```python
conf = clamp((score - min_top20_score) / (max_top20_score - min_top20_score), 0, 1)
```

## two-router 训练脚本
已建：
- `run_two_router_multiview_layered.sh`

当前指向：
- train：`train_qa_pairs_kqt_candidates_target_poi_multiview_dst_time_layered12_far3_mismatch3_tail2.json`
- test：`test_qa_pairs_kqt_candidates_target_poi_multiview_dst_time_layered_top20.txt`
- output：`output/two-router-target-poi-multiview-dst-time-layered12-far3-mismatch3-tail2`

## retriever hard negative loss
已在 `rag/target_poi_multiview/train.py` 增加：
- `--rank_weight`（默认 0）
- `--rank_margin`（默认 0.1）
- `--hard_topk`（默认 20）

损失为 fused query 上的 margin rank loss（与原 CE 并行）。

## 当前最关键目标
最终优化目标是 LLM 最终 Acc，不是 retriever 单独 Recall@20。
优先做“候选构造 + prompt 信息”优化，再看是否继续加强 retriever loss。
