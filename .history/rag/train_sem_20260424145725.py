import os
import re
import json
import math
import bisect
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from poi_data_loader import load_full_dataset
from transition_graph import TimeBucketManager


class SemanticEncoder(nn.Module):
    def __init__(self, llm_dim=768, num_buckets=8, hidden=512, shared_dim=128, dropout=0.2):
        super().__init__()
        self.shared_dim = shared_dim
        self.bucket_emb = nn.Embedding(num_buckets, 32)

        self.query_mlp = nn.Sequential(
            nn.Linear(llm_dim + 32, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, shared_dim),
        )

        self.candidate_mlp = nn.Sequential(
            nn.Linear(llm_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, shared_dim),
        )

    def encode_query(self, llm_vec, bucket_ids):
        h_bucket = self.bucket_emb(bucket_ids)
        x = torch.cat([llm_vec, h_bucket], dim=-1)
        return F.normalize(self.query_mlp(x), dim=-1)

    def encode_candidate(self, llm_vec):
        return F.normalize(self.candidate_mlp(llm_vec), dim=-1)


def parse_qa_sample(question_text):
    trajectory = []
    target_epoch = None

    pattern_a = r"\[Current trajectory'?s? check-in sequence\]:\s*(.*?)\s*\[Historical check-in sequences\]:"
    m = re.search(pattern_a, question_text, re.DOTALL)

    if m:
        traj_text = m.group(1)
    else:
        pattern_b = r"\[Current trajectory'?s? check-in sequence\]:\s*(.*?)\s*Given the data,"
        m2 = re.search(pattern_b, question_text, re.DOTALL)
        if m2:
            traj_text = m2.group(1)
        else:
            parts = question_text.split("Given the data,")
            traj_text = parts[0] if parts else question_text

    visit_pattern = (
        r"At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+"
        r"user\s+\d+\s+visited\s+POI\s+id\s+(\d+)\s+"
        r"which\s+is\s+a\s+(.+?)\s+with\s+Category\s+id\s+(\d+)"
    )

    for item in re.finditer(visit_pattern, traj_text):
        time_str, poi_id, category_name, category_id = item.groups()
        try:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            trajectory.append(
                {
                    "poi_id": int(poi_id),
                    "category": category_name.strip(),
                    "category_id": int(category_id),
                    "epoch": int(dt.timestamp()),
                    "hour": dt.hour,
                    "day_of_week": dt.weekday(),
                }
            )
        except Exception:
            continue

    target_pattern = r"Given the data,\s+At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+Which"
    t = re.search(target_pattern, question_text)
    if t:
        try:
            dt = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")
            target_epoch = int(dt.timestamp())
        except Exception:
            target_epoch = None

    return trajectory, target_epoch


def parse_answer_poi(answer_text):
    m = re.search(r"POI\s+id\s+(\d+)", answer_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b", answer_text)
    if m:
        return int(m.group(1))
    return None


def load_qa_data(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if not txt:
        return []

    if path.lower().endswith(".json"):
        data = json.loads(txt)
        if isinstance(data, list):
            return data
        raise ValueError("JSON file must contain a list")

    # .txt may be jsonl or a full json list
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


class TemporalTargetIndex:
    """Index facts by (bucket, target_poi_idx) -> sorted (epoch, source_poi_idx)."""

    def __init__(self):
        self.index = defaultdict(list)
        self._epochs_cache = {}

    def add(self, bucket, target_idx, epoch, source_idx):
        self.index[(bucket, target_idx)].append((epoch, source_idx))

    def build(self):
        for key in self.index:
            self.index[key].sort(key=lambda x: x[0])
            self._epochs_cache[key] = [e for e, _ in self.index[key]]

    def positives_before(self, bucket, target_idx, cutoff_epoch):
        entries = self.index.get((bucket, target_idx), [])
        if not entries:
            return {}

        epochs = self._epochs_cache[(bucket, target_idx)]
        n = bisect.bisect_left(epochs, cutoff_epoch)
        counts = defaultdict(int)
        for i in range(n):
            counts[entries[i][1]] += 1
        return dict(counts)

    def positives_all(self, bucket, target_idx):
        entries = self.index.get((bucket, target_idx), [])
        counts = defaultdict(int)
        for _, src in entries:
            counts[src] += 1
        return dict(counts)


def _get_train_trajectories(dataset):
    for attr in ["train_trajectories", "train_trajs", "trajectories_train"]:
        val = getattr(dataset, attr, None)
        if isinstance(val, dict):
            print(f"Using dataset.{attr} as train fact source")
            return val, attr
    print("dataset.train_trajectories not found; fallback to dataset.all_trajectories")
    return dataset.all_trajectories, "all_trajectories"


def build_fact_index_from_train(dataset, p2i, tm):
    trajectories, source_attr = _get_train_trajectories(dataset)
    idx = TemporalTargetIndex()

    for traj in trajectories.values():
        visits = getattr(traj, "visits", [])
        for i in range(len(visits) - 1):
            s = visits[i]
            d = visits[i + 1]

            si = p2i.get(s.poi_id, -1)
            di = p2i.get(d.poi_id, -1)
            if si < 0 or di < 0:
                continue

            b = tm.get_bucket_from_visit(s)
            idx.add(bucket=b, target_idx=di, epoch=s.epoch, source_idx=si)

    idx.build()
    print(f"Fact index built from {source_attr}: {len(idx.index)} keys")
    return idx, source_attr


def build_samples_from_qa(qa_path, p2i, tm):
    qa_data = load_qa_data(qa_path)
    samples = []

    for i, item in enumerate(qa_data):
        q = item.get("question", "")
        a = item.get("answer", "")

        traj, target_epoch = parse_qa_sample(q)
        target_poi = parse_answer_poi(a)

        if not traj or target_epoch is None or target_poi is None:
            continue

        last = traj[-1]
        query_poi = last["poi_id"]
        query_idx = p2i.get(query_poi, -1)
        target_idx = p2i.get(target_poi, -1)
        if query_idx < 0 or target_idx < 0:
            continue

        hour = last["hour"]
        is_weekend = last["day_of_week"] >= 5
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
    print(f"Parsed {len(samples)} valid samples from {qa_path}")
    return samples


def attach_positives(samples, fact_index, max_pos=32, drop_zero=True, use_time_mask=True):
    out = []

    for s in samples:
        if use_time_mask:
            counts = fact_index.positives_before(
                bucket=s["bucket"], target_idx=s["target_poi_idx"], cutoff_epoch=s["target_epoch"]
            )
        else:
            counts = fact_index.positives_all(bucket=s["bucket"], target_idx=s["target_poi_idx"])

        if not counts:
            if drop_zero:
                continue
            s2 = dict(s)
            s2["pos_indices"] = []
            s2["pos_weights"] = []
            out.append(s2)
            continue

        # Keep strongest positives first, then cap for stable compute.
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if max_pos is not None and len(items) > max_pos:
            head = items[: max_pos // 2]
            tail = items[max_pos // 2 :]
            np.random.shuffle(tail)
            items = head + tail[: (max_pos - len(head))]

        pos_indices = [idx for idx, _ in items]
        w = np.array([math.log1p(c) for _, c in items], dtype=np.float32)
        w = w / (w.sum() + 1e-8)

        s2 = dict(s)
        s2["pos_indices"] = pos_indices
        s2["pos_weights"] = w.tolist()
        out.append(s2)

    print(f"Samples after attaching positives: {len(out)}")
    return out


class QASemanticDataset(torch.utils.data.Dataset):
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
        "pos_indices": [x["pos_indices"] for x in batch],
        "pos_weights": [x["pos_weights"] for x in batch],
    }


def compute_multipos_loss(scores, pos_indices, pos_weights):
    """Listwise CE over a multi-positive distribution."""
    bsz, n = scores.shape
    log_probs = F.log_softmax(scores, dim=1)
    target = torch.zeros_like(scores)

    for i in range(bsz):
        idxs = pos_indices[i]
        ws = pos_weights[i]
        if not idxs:
            continue
        idx_t = torch.tensor(idxs, dtype=torch.long, device=scores.device)
        w_t = torch.tensor(ws, dtype=torch.float32, device=scores.device)
        target[i, idx_t] = w_t

    row_sum = target.sum(dim=1)
    valid = row_sum > 0
    if valid.sum().item() == 0:
        return None

    per_row = -(target[valid] * log_probs[valid]).sum(dim=1)
    return per_row.mean()


def compute_rank_hinge(scores, pos_indices, margin=0.15, hard_neg_k=8):
    bsz, n = scores.shape
    losses = []

    for i in range(bsz):
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
        l = F.relu(margin - pos_score + hard_vals).mean()
        losses.append(l)

    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def evaluate_coverage(samples, sem_model, sem_all, top_ks, device, count_topk=10):
    sem_model.eval()
    dev = torch.device(device)

    with torch.no_grad():
        cand = sem_model.encode_candidate(sem_all)

    hit = {k: 0 for k in top_ks}
    pos_at_least_3 = 0
    pos_at_least_5 = 0
    total = 0

    for s in samples:
        if not s["pos_indices"]:
            continue

        qi = torch.tensor([s["query_poi_idx"]], dtype=torch.long, device=dev)
        b = torch.tensor([s["bucket"]], dtype=torch.long, device=dev)

        with torch.no_grad():
            zq = sem_model.encode_query(sem_all[qi], b)
            scores = (zq @ cand.t()).squeeze(0)

        max_k = max(top_ks)
        top_ids = torch.topk(scores, k=min(max_k, scores.numel())).indices.tolist()
        pos_set = set(s["pos_indices"])

        for k in top_ks:
            ids = top_ids[:k]
            if any(i in pos_set for i in ids):
                hit[k] += 1

        ids_k = top_ids[:count_topk]
        n_pos_in_topk = sum(1 for i in ids_k if i in pos_set)
        if n_pos_in_topk >= 3:
            pos_at_least_3 += 1
        if n_pos_in_topk >= 5:
            pos_at_least_5 += 1
        total += 1

    if total == 0:
        out = {f"coverage@{k}": 0.0 for k in top_ks}
        out[f"pos>=3@{count_topk}"] = 0.0
        out[f"pos>=5@{count_topk}"] = 0.0
        return out

    out = {f"coverage@{k}": hit[k] / total for k in top_ks}
    out[f"pos>=3@{count_topk}"] = pos_at_least_3 / total
    out[f"pos>=5@{count_topk}"] = pos_at_least_5 / total
    return out


def plot_training_curves(history, save_path):
    if not history:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is unavailable; skip loss curve plotting")
        return

    epochs = [h["epoch"] for h in history]
    total_loss = [h.get("loss", 0.0) for h in history]
    main_loss = [h.get("loss_main", 0.0) for h in history]
    rank_loss = [h.get("loss_rank", 0.0) for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_loss, label="Total loss", linewidth=2.2)
    plt.plot(epochs, main_loss, label="Main loss", linewidth=1.8)
    plt.plot(epochs, rank_loss, label="Rank loss", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Semantic Retriever Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def train_semantic(
    dataset,
    tm,
    train_qa,
    val_qa,
    save_dir="./sem_lib/",
    encoder_name="bert",
    epochs=120,
    batch_size=128,
    lr=3e-4,
    patience=12,
    tau=0.1,
    shared_dim=128,
    lambda_rank=0.10,
    max_pos=32,
    hard_neg_k=8,
    device="cpu",
):
    os.makedirs(save_dir, exist_ok=True)
    dev = torch.device(device)

    all_pois = sorted(dataset.poi_dict.keys())
    p2i = {p: i for i, p in enumerate(all_pois)}
    n_pois = len(all_pois)

    print("Loading semantic features...")
    try:
        from train_semantic_view import prepare_features

        fd = prepare_features(dataset, None, all_pois, encoder_name, device, True)
    except Exception:
        sem_v = np.random.randn(n_pois, 32).astype(np.float32)
        fd = {"sem_vectors": sem_v, "llm_dim": 32}

    sem_all = torch.tensor(fd["sem_vectors"], dtype=torch.float32, device=dev)
    llm_dim = int(fd["llm_dim"])

    print("Building train fact index...")
    fact_index, fact_source_attr = build_fact_index_from_train(dataset, p2i, tm)

    print("Building train samples (using all positives - no time mask for representation learning)...")
    raw_train_samples = build_samples_from_qa(train_qa, p2i, tm)
    train_samples = attach_positives(
        raw_train_samples, fact_index, max_pos=max_pos, drop_zero=True, use_time_mask=False
    )

    print("Building validate samples (using all positives - no time mask)...")
    raw_val_samples = build_samples_from_qa(val_qa, p2i, tm)
    val_samples = attach_positives(
        raw_val_samples, fact_index, max_pos=None, drop_zero=False, use_time_mask=False
    )

    if len(train_samples) < 100:
        raise RuntimeError("Too few train samples after filtering; check QA parsing/index rules.")
    if len(val_samples) == 0:
        raise RuntimeError("No validate samples parsed; check --val_qa path/format.")

    train_ds = QASemanticDataset(train_samples)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = SemanticEncoder(llm_dim=llm_dim, num_buckets=tm.num_buckets, hidden=512, shared_dim=shared_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_metric = -1.0
    wait = 0
    history = []

    val_nonzero = sum(1 for s in val_samples if s["pos_indices"])
    print(
        f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} "
        f"(non-empty positives: {val_nonzero}) | POIs: {n_pois}"
    )

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_main = 0.0
        running_rank = 0.0
        n_batch = 0

        for batch in train_loader:
            opt.zero_grad()

            qidx = batch["query_poi_idx"].to(dev)
            buck = batch["bucket"].to(dev)
            cand = model.encode_candidate(sem_all)
            zq = model.encode_query(sem_all[qidx], buck)
            scores = (zq @ cand.t()) / tau

            loss_main = compute_multipos_loss(scores, batch["pos_indices"], batch["pos_weights"])
            if loss_main is None:
                continue

            loss_rank = compute_rank_hinge(scores, batch["pos_indices"], margin=0.15, hard_neg_k=hard_neg_k)
            loss = loss_main + lambda_rank * loss_rank

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            running_main += loss_main.item()
            running_rank += loss_rank.item()
            n_batch += 1

        sched.step()
        avg_loss = running / max(1, n_batch)
        avg_main_loss = running_main / max(1, n_batch)
        avg_rank_loss = running_rank / max(1, n_batch)

        val_cov = evaluate_coverage(
            val_samples, model, sem_all, top_ks=(1, 5, 10, 20, 50), device=device, count_topk=10
        )
        key_metric = val_cov["coverage@20"]

        history.append(
            {
                "epoch": ep,
                "loss": avg_loss,
                "loss_main": avg_main_loss,
                "loss_rank": avg_rank_loss,
                **val_cov,
            }
        )

        print(
            f"E{ep:03d} | loss={avg_loss:.4f} (main={avg_main_loss:.4f}, rank={avg_rank_loss:.4f}) "
            f"| cov@10={val_cov['coverage@10']:.4f} | cov@20={val_cov['coverage@20']:.4f} "
            f"| cov@50={val_cov['coverage@50']:.4f} | pos>=3@10={val_cov['pos>=3@10']:.4f} "
            f"| pos>=5@10={val_cov['pos>=5@10']:.4f}"
        )

        if key_metric > best_metric + 1e-4:
            best_metric = key_metric
            wait = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "sem_encoder.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    model.load_state_dict(torch.load(os.path.join(save_dir, "sem_encoder.pth"), map_location=dev))
    model.eval()
    with torch.no_grad():
        poi_embs = model.encode_candidate(sem_all).cpu().numpy()

    np.save(os.path.join(save_dir, "poi_embeddings.npy"), poi_embs)
    with open(os.path.join(save_dir, "poi_id_list.json"), "w", encoding="utf-8") as f:
        json.dump(all_pois, f)

    config = {
        "shared_dim": shared_dim,
        "llm_dim": llm_dim,
        "num_buckets": tm.num_buckets,
        "num_pois": n_pois,
        "tau": tau,
        "max_pos": max_pos,
        "hard_neg_k": hard_neg_k,
        "lambda_rank": lambda_rank,
        "best_val_coverage20": best_metric,
        "fact_source": fact_source_attr,
        "train_qa": train_qa,
        "val_qa": val_qa,
        "val_positive_rule": "full_time_no_mask",
    }
    torch.save(config, os.path.join(save_dir, "config.pth"))

    with open(os.path.join(save_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    curve_path = os.path.join(save_dir, "loss_curve.png")
    plot_training_curves(history, curve_path)
    if os.path.exists(curve_path):
        print(f"Saved loss curve to {curve_path}")

    print(f"Training done. Best coverage@20={best_metric:.4f}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="./datasets/nyc/preprocessed/train_sample.csv")
    p.add_argument("--test_csv", default="./datasets/nyc/preprocessed/test_sample.csv")
    p.add_argument("--train_qa", default="./datasets/nyc/preprocessed/train_qa_pairs_kqt.json")
    p.add_argument(
        "--val_qa",
        default="./datasets/nyc/preprocessed/validate_qa_pairs_kqt_history.json",
    )
    p.add_argument("--save_dir", default="./sem_lib/")
    p.add_argument("--encoder", default="bert")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--shared_dim", type=int, default=128)
    p.add_argument("--max_pos", type=int, default=32)
    p.add_argument("--hard_neg_k", type=int, default=8)
    p.add_argument("--lambda_rank", type=float, default=0.20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ds = load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.val_qa)
    tm = TimeBucketManager()

    train_semantic(
        dataset=ds,
        tm=tm,
        train_qa=args.train_qa,
        val_qa=args.val_qa,
        save_dir=args.save_dir,
        encoder_name=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        tau=args.tau,
        shared_dim=args.shared_dim,
        lambda_rank=args.lambda_rank,
        max_pos=args.max_pos,
        hard_neg_k=args.hard_neg_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()
