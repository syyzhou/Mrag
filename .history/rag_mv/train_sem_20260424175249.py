import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
RAG_DIR = os.path.join(ROOT_DIR, "rag")
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from rag.transition_graph import TimeBucketManager
from rag_mv.data import build_poi_mappings, load_mv_dataset, parse_answer_poi, parse_qa_sample
from rag_mv.features import prepare_poi_features


class SemanticEncoder(nn.Module):
    def __init__(self, llm_dim=768, num_buckets=8, hidden_dim=512, shared_dim=128, dropout=0.2):
        super().__init__()
        self.shared_dim = shared_dim
        self.bucket_emb = nn.Embedding(num_buckets, 32)

        self.query_mlp = nn.Sequential(
            nn.Linear(llm_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
        )

        self.candidate_mlp = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
        )

    def encode_query(self, llm_vec, bucket_ids):
        h_bucket = self.bucket_emb(bucket_ids)
        x = torch.cat([llm_vec, h_bucket], dim=-1)
        return F.normalize(self.query_mlp(x), dim=-1)

    def encode_candidate(self, llm_vec):
        return F.normalize(self.candidate_mlp(llm_vec), dim=-1)


class TemporalTargetIndex:
    """(bucket, target_idx) -> sorted [(epoch, source_idx), ...]."""

    def __init__(self):
        self.index = defaultdict(list)
        self._epochs_cache = {}

    def add(self, bucket, target_idx, epoch, source_idx):
        self.index[(int(bucket), int(target_idx))].append((int(epoch), int(source_idx)))

    def build(self):
        for key in self.index:
            self.index[key].sort(key=lambda x: x[0])
            self._epochs_cache[key] = [e for e, _ in self.index[key]]

    def positives_before(self, bucket, target_idx, cutoff_epoch):
        entries = self.index.get((int(bucket), int(target_idx)), [])
        if not entries:
            return {}
        epochs = self._epochs_cache[(int(bucket), int(target_idx))]
        n = np.searchsorted(epochs, int(cutoff_epoch), side="left")
        counts = defaultdict(int)
        for i in range(n):
            counts[entries[i][1]] += 1
        return dict(counts)

    def positives_all(self, bucket, target_idx):
        entries = self.index.get((int(bucket), int(target_idx)), [])
        counts = defaultdict(int)
        for _, src in entries:
            counts[src] += 1
        return dict(counts)


def build_fact_index_from_train(dataset, p2i, tm):
    idx = TemporalTargetIndex()
    trajectories = dataset.all_trajectories

    for traj in trajectories.values():
        visits = getattr(traj, "visits", [])
        for i in range(len(visits) - 1):
            src = visits[i]
            dst = visits[i + 1]
            src_idx = p2i.get(int(src.poi_id), -1)
            dst_idx = p2i.get(int(dst.poi_id), -1)
            if src_idx < 0 or dst_idx < 0:
                continue
            bucket = tm.get_bucket_from_visit(src)
            idx.add(bucket=bucket, target_idx=dst_idx, epoch=int(src.epoch), source_idx=src_idx)

    idx.build()
    return idx


def build_samples_from_qa(qa_pairs, p2i, tm):
    samples = []
    for i, qa in enumerate(qa_pairs):
        q = qa.question
        a = qa.answer

        traj, target_epoch, _ = parse_qa_sample(q)
        target_poi = parse_answer_poi(a)
        if not traj or target_epoch is None or target_poi is None:
            continue

        last = traj[-1]
        query_poi = int(last["poi_id"])
        query_idx = p2i.get(query_poi, -1)
        target_idx = p2i.get(int(target_poi), -1)
        if query_idx < 0 or target_idx < 0:
            continue

        hour = int(last["hour"])
        is_weekend = int(last["day_of_week"]) >= 5
        bucket = tm.get_bucket(hour, is_weekend)
        samples.append(
            {
                "sample_id": i,
                "query_poi_idx": query_idx,
                "target_poi_idx": target_idx,
                "bucket": int(bucket),
                "target_epoch": int(target_epoch),
            }
        )

    samples.sort(key=lambda x: x["target_epoch"])
    return samples


def attach_positives(samples, fact_index, use_time_mask=True, max_pos=32, drop_zero=True):
    out = []
    for sample in samples:
        counts = (
            fact_index.positives_before(sample["bucket"], sample["target_poi_idx"], sample["target_epoch"])
            if use_time_mask
            else fact_index.positives_all(sample["bucket"], sample["target_poi_idx"])
        )

        if not counts:
            if drop_zero:
                continue
            sample2 = dict(sample)
            sample2["pos_indices"] = []
            sample2["pos_weights"] = []
            out.append(sample2)
            continue

        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if max_pos is not None and len(items) > max_pos:
            items = items[:max_pos]

        pos_indices = [idx for idx, _ in items]
        weights = np.array([math.log1p(c) for _, c in items], dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)

        sample2 = dict(sample)
        sample2["pos_indices"] = pos_indices
        sample2["pos_weights"] = weights.tolist()
        out.append(sample2)
    return out


class SemanticTrainDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "query_poi_idx": torch.tensor([x["query_poi_idx"] for x in batch], dtype=torch.long),
        "bucket": torch.tensor([x["bucket"] for x in batch], dtype=torch.long),
        "target_poi_idx": torch.tensor([x["target_poi_idx"] for x in batch], dtype=torch.long),
        "pos_indices": [x["pos_indices"] for x in batch],
        "pos_weights": [x["pos_weights"] for x in batch],
    }


def compute_multipos_loss(scores, pos_indices, pos_weights):
    log_probs = F.log_softmax(scores, dim=1)
    target = torch.zeros_like(scores)

    for i in range(scores.size(0)):
        idxs = pos_indices[i]
        ws = pos_weights[i]
        if not idxs:
            continue
        idx_t = torch.tensor(idxs, dtype=torch.long, device=scores.device)
        w_t = torch.tensor(ws, dtype=torch.float32, device=scores.device)
        target[i, idx_t] = w_t

    valid = target.sum(dim=1) > 0
    if valid.sum().item() == 0:
        return None
    return -(target[valid] * log_probs[valid]).sum(dim=1).mean()


def compute_rank_hinge(scores, pos_indices, margin=0.15, hard_neg_k=8):
    losses = []
    n = scores.size(1)
    for i in range(scores.size(0)):
        pos = pos_indices[i]
        if not pos:
            continue
        pos_t = torch.tensor(pos, dtype=torch.long, device=scores.device)
        pos_score = scores[i, pos_t].mean()
        neg_mask = torch.ones(n, dtype=torch.bool, device=scores.device)
        neg_mask[pos_t] = False
        neg_scores = scores[i].masked_fill(~neg_mask, float("-inf"))
        k = min(hard_neg_k, int(neg_mask.sum().item()))
        if k <= 0:
            continue
        hard_vals = torch.topk(neg_scores, k=k).values
        losses.append(F.relu(margin - pos_score + hard_vals).mean())
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def evaluate_coverage(samples, model, sem_all, top_ks, device):
    dev = torch.device(device)
    model.eval()
    with torch.no_grad():
        cand = model.encode_candidate(sem_all)

    hit = {k: 0 for k in top_ks}
    total = 0
    for sample in samples:
        if not sample["pos_indices"]:
            continue
        qi = torch.tensor([sample["query_poi_idx"]], dtype=torch.long, device=dev)
        bucket = torch.tensor([sample["bucket"]], dtype=torch.long, device=dev)
        with torch.no_grad():
            zq = model.encode_query(sem_all[qi], bucket)
            scores = (zq @ cand.t()).squeeze(0)

        top_ids = torch.topk(scores, k=min(max(top_ks), scores.numel())).indices.tolist()
        pos_set = set(sample["pos_indices"])
        for k in top_ks:
            if any(idx in pos_set for idx in top_ids[:k]):
                hit[k] += 1
        total += 1

    if total == 0:
        return {f"coverage@{k}": 0.0 for k in top_ks}
    return {f"coverage@{k}": hit[k] / total for k in top_ks}


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = load_mv_dataset(args.train_csv, args.test_csv, args.train_qa, args.val_qa)
    tm = TimeBucketManager()
    all_pois, p2i = build_poi_mappings(dataset)

    feature_dict = prepare_poi_features(dataset, all_pois, args.encoder, args.device)
    sem_all = torch.tensor(feature_dict["sem_vectors"], dtype=torch.float32, device=device)
    llm_dim = int(feature_dict["llm_dim"])

    fact_index = build_fact_index_from_train(dataset, p2i, tm)
    train_samples = attach_positives(
        build_samples_from_qa(dataset.train_qa_pairs, p2i, tm),
        fact_index,
        use_time_mask=True,
        max_pos=args.max_pos,
        drop_zero=True,
    )
    val_samples = attach_positives(
        build_samples_from_qa(dataset.test_qa_pairs, p2i, tm),
        fact_index,
        use_time_mask=False,
        max_pos=args.max_pos,
        drop_zero=False,
    )

    if not train_samples:
        raise RuntimeError("No valid training samples for semantic view.")

    loader = DataLoader(
        SemanticTrainDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = SemanticEncoder(
        llm_dim=llm_dim,
        num_buckets=tm.num_buckets,
        hidden_dim=args.hidden_dim,
        shared_dim=args.shared_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    history = []
    best_metric = -1.0
    wait = 0

    print(f"[sem] train_samples={len(train_samples)} val_samples={len(val_samples)} num_pois={len(all_pois)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        running_main = 0.0
        running_rank = 0.0
        n_batch = 0

        for batch in loader:
            optimizer.zero_grad()
            qidx = batch["query_poi_idx"].to(device)
            bucket = batch["bucket"].to(device)

            cand = model.encode_candidate(sem_all)
            zq = model.encode_query(sem_all[qidx], bucket)
            scores = (zq @ cand.t()) / max(args.tau, 1e-6)

            loss_main = compute_multipos_loss(scores, batch["pos_indices"], batch["pos_weights"])
            if loss_main is None:
                continue
            loss_rank = compute_rank_hinge(scores, batch["pos_indices"], margin=args.margin, hard_neg_k=args.hard_neg_k)
            loss = loss_main + args.lambda_rank * loss_rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += float(loss.item())
            running_main += float(loss_main.item())
            running_rank += float(loss_rank.item())
            n_batch += 1

        scheduler.step()
        metrics = evaluate_coverage(val_samples, model, sem_all, top_ks=(5, 10, 20, 50), device=args.device)
        avg_loss = running / max(1, n_batch)
        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "loss_main": running_main / max(1, n_batch),
            "loss_rank": running_rank / max(1, n_batch),
            **metrics,
        }
        history.append(row)
        print(
            f"[sem] epoch={epoch} loss={row['loss']:.4f} main={row['loss_main']:.4f} "
            f"rank={row['loss_rank']:.4f} cov@20={row['coverage@20']:.4f}"
        )

        key_metric = row["coverage@20"]
        if key_metric > best_metric + 1e-4:
            best_metric = key_metric
            wait = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "sem_encoder.pth"))
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[sem] early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(os.path.join(args.save_dir, "sem_encoder.pth"), map_location=device))
    model.eval()
    with torch.no_grad():
        poi_embs = model.encode_candidate(sem_all).cpu().numpy()

    np.save(os.path.join(args.save_dir, "poi_embeddings.npy"), poi_embs)
    with open(os.path.join(args.save_dir, "poi_id_list.json"), "w", encoding="utf-8") as f:
        json.dump(all_pois, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    torch.save(
        {
            "encoder_name": args.encoder,
            "llm_dim": llm_dim,
            "shared_dim": args.shared_dim,
            "hidden_dim": args.hidden_dim,
            "num_buckets": tm.num_buckets,
            "tau": args.tau,
            "lambda_rank": args.lambda_rank,
            "hard_neg_k": args.hard_neg_k,
            "max_pos": args.max_pos,
            "best_val_coverage20": best_metric,
        },
        os.path.join(args.save_dir, "config.pth"),
    )


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="./datasets/nyc/preprocessed/train_sample.csv")
    p.add_argument("--test_csv", default="./datasets/nyc/preprocessed/test_sample.csv")
    p.add_argument("--train_qa", default="./datasets/nyc/preprocessed/train_qa_pairs_kqt.json")
    p.add_argument("--val_qa", default="./datasets/nyc/preprocessed/test_qa_pairs_kqt.json")
    p.add_argument("--save_dir", default="./rag_mv/artifacts/sem/")
    p.add_argument("--encoder", default="bert")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--shared_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lambda_rank", type=float, default=0.2)
    p.add_argument("--hard_neg_k", type=int, default=8)
    p.add_argument("--max_pos", type=int, default=32)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
