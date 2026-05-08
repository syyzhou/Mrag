# Target POI Multi-View Retrieval

This is an isolated baseline for direct target-POI retrieval:

```text
semantic view   \
structure view   -> gated fusion -> target POI retrieval
trajectory view /
```

The model uses one shared POI encoder for target POI embeddings. Each POI
embedding is built from:

- trainable POI id embedding
- cached LLM/text/category vector projection from `rag/feature_cache/bert`
- cached geographic feature projection

Views:

- `semantic`: last observed POI embedding + target/current time embedding
- `trajectory`: Transformer encoder over the observed POI sequence
- `structure`: GAT over a transition subgraph centered at the last observed POI

Training objective:

- main CE loss: fused query retrieves the true `target_poi_id`
- auxiliary CE losses: semantic/structure/trajectory queries retrieve the target
- optional lightweight view alignment loss

Example smoke run:

```bash
DATASET_NAME=nyc
python rag/target_poi_multiview/train.py \
  --dataset_name ${DATASET_NAME} \
  --train_csv ./datasets/${DATASET_NAME}/preprocessed/train_sample.csv \
  --test_csv ./datasets/${DATASET_NAME}/preprocessed/test_sample_with_traj.csv \
  --train_qa ./datasets/${DATASET_NAME}/preprocessed/train_qa_pairs_kqt.json \
  --test_qa ./datasets/${DATASET_NAME}/preprocessed/test_qa_pairs_kqt.json \
  --feature_cache ./rag/feature_cache/bert \
  --max_train_samples 512 \
  --max_val_samples 256 \
  --epochs 1 \
  --device cpu
```

First-stage baseline should keep `--align_weight 0`. After Recall@20 is stable,
try `--align_weight 0.01` to `0.03`.
