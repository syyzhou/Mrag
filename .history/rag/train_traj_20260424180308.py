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

from poi_data_loader import load_full_dataset


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        llm_dim=768,
        geo_dim=6,
        hour_emb_dim=16,
        dow_emb_dim=16,
        hidden=128,
        shared_dim=128,
        n_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.hour_emb = nn.Embedding(24, hour_emb_dim)
        self.dow_emb = nn.Embedding(7, dow_emb_dim)
        tdim = hour_emb_dim + dow_emb_dim

        self.poi_proj = nn.Sequential(nn.Linear(llm_dim, hidden), nn.ReLU())
        self.geo_proj = nn.Sequential(nn.Linear(geo_dim, hidden), nn.ReLU())
        self.time_proj = nn.Sequential(nn.Linear(tdim, hidden), nn.ReLU())

        self.gru = nn.GRU(
            hidden * 3,
            hidden,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out_proj = nn.Sequential(nn.Linear(hidden, shared_dim), nn.LayerNorm(shared_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, llm_seq, geo_seq, hours, dows, lengths):
        h_poi = self.poi_proj(llm_seq)
        h_geo = self.geo_proj(geo_seq)
        h_time = self.time_proj(torch.cat([self.hour_emb(hours), self.dow_emb(dows)], dim=-1))

        x = self.dropout(torch.cat([h_poi, h_geo, h_time], dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        return F.normalize(self.out_proj(h_n[-1]), dim=-1)


class TrajectoryTargetIndex:
    """
    target_poi_id -> sorted [(epoch, pool_idx), ...]
    """

    def __init__(self):
        self.by_target = defaultdict(list)
        self._epoch_cache = {}

    def add(self, target_poi_id, epoch, pool_idx):
        self.by_target[target_poi_id].append((epoch, pool_idx))

    def build(self):
        for key in self.by_target:
            self.by_target[key].sort(key=lambda x: x[0])
            self._epoch_cache[key] = [e for e, _ in self.by_target[key]]

    def positives_before(self, target_poi_id, cutoff_epoch):
        entries = self.by_target.get(target_poi_id, [])
        if not entries:
            return {}
        epochs = self._epoch_cache[target_poi_id]
        n = bisect.bisect_left(epochs, cutoff_epoch)
        counts = defaultdict(int)
        for i in range(n):
            _, pool_idx = entries[i]
            counts[pool_idx] += 1
        return dict(counts)

    def positives_all(self, target_poi_id):
        entries = self.by_target.get(target_poi_id, [])
        counts = defaultdict(int)
        for _, pool_idx in entries:
            counts[pool_idx] += 1
        return dict(counts)


def parse_qa_sample(question_text):
    trajectory = []
    target_epoch = None

    pa = r"\[Current trajectory'?s? check-in sequence\]:\s*(.*?)\s*\[Historical check-in sequences\]:"
    m = re.search(pa, question_text, re.DOTALL)
    if m:
        traj_text = m.group(1)
    else:
        pb = r"\[Current trajectory'?s? check-in sequence\]:\s*(.*?)\s*Given the data,"
        m2 = re.search(pb, question_text, re.DOTALL)
        if m2:
            traj_text = m2.group(1)
        else:
            traj_text = question_text.split("Given the data,")[0]

    visit_pattern = (
        r"At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+"
        r"user\s+\d+\s+visited\s+POI\s+id\s+(\d+)\s+"
        r"which\s+is\s+a\s+(.+?)\s+with\s+Category\s+id\s+(\d+)"
    )

    for item in re.finditer(visit_pattern, traj_text):
        t_str, poi_id, cat_name, cat_id = item.groups()
        try:
            dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
            trajectory.append(
                {
                    "poi_id": int(poi_id),
                    "epoch": int(dt.timestamp()),
                    "category": cat_name.strip(),
                    "category_id": int(cat_id),
                    "hour": dt.hour,
                    "day_of_week": dt.weekday(),
                }
            )
        except Exception:
            continue

    tp = r"Given the data,\s+At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+Which"
    tm = re.search(tp, question_text)
    if tm:
        try:
            dt = datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S")
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


def _get_train_trajectories(dataset):
    for attr in ["train_trajectories", "train_trajs", "trajectories_train"]:
        val = getattr(dataset, attr, None)
        if isinstance(val, dict):
            print(f"Using dataset.{attr} as trajectory pool")
            return val, attr
    print("dataset.train_trajectories not found; fallback to dataset.all_trajectories")
    return dataset.all_trajectories, "all_trajectories(fallback)"


def visits_to_record(visits, p2i):
    pis = [p2i.get(int(v["poi_id"]), 0) for v in visits]
    hs = [int(v.get("hour", 0)) % 24 for v in visits]
    ds = [int(v.get("day_of_week", 0)) % 7 for v in visits]
    if not pis:
        pis, hs, ds = [0], [0], [0]
    return {
        "pis": torch.tensor(pis, dtype=torch.long),
        "hs": torch.tensor(hs, dtype=torch.long),
        "ds": torch.tensor(ds, dtype=torch.long),
        "l": len(pis),
    }


def encode_records(records, model, sem_all, geo_all, dev, shared_dim):
    if not records:
        return torch.empty(0, shared_dim, device=dev)
    max_l = max(r["l"] for r in records)
    bp = torch.stack([F.pad(r["pis"], (0, max_l - r["l"])) for r in records]).to(dev)
    bh = torch.stack([F.pad(r["hs"], (0, max_l - r["l"])) for r in records]).to(dev)
    bd = torch.stack([F.pad(r["ds"], (0, max_l - r["l"])) for r in records]).to(dev)
    bl = torch.tensor([r["l"] for r in records], dtype=torch.long, device=dev)
    return model(sem_all[bp], geo_all[bp], bh, bd, bl)


def build_pool(dataset, p2i):
    traj_dict, source_attr = _get_train_trajectories(dataset)
    pool_records = []
    pool_meta = []
    index = TrajectoryTargetIndex()

    for traj_id, traj in traj_dict.items():
        visits = getattr(traj, "visits", [])
        if not visits:
            continue

        vis_list = []
        for v in visits:
            vis_list.append(
                {
                    "poi_id": int(v.poi_id),
                    "epoch": int(v.epoch),
                    "hour": int(v.hour),
                    "day_of_week": int(v.day_of_week),
                    "category_name": getattr(v, "category_name", "Unknown"),
                    "category_id": int(getattr(v, "category_id", 0)),
                }
            )

        pool_idx = len(pool_records)
        pool_records.append(visits_to_record(vis_list, p2i))
        pool_meta.append(
            {
                "pool_idx": pool_idx,
                "traj_id": traj_id,
                "user_id": int(getattr(traj, "user_id", -1)),
                "last_epoch": int(vis_list[-1]["epoch"]),
                "traj_len": len(vis_list),
            }
        )

        # Use the trajectory endpoint as the positive target.
        # Indexing every intermediate POI makes supervision too noisy for top-k retrieval.
        index.add(target_poi_id=vis_list[-1]["poi_id"], epoch=vis_list[-1]["epoch"], pool_idx=pool_idx)

    index.build()
    print(f"Trajectory pool built: {len(pool_records)} records")
    return pool_records, pool_meta, index, source_attr


def build_samples_from_qa(path, p2i):
    qa_data = load_qa_data(path)
    samples = []

    for i, row in enumerate(qa_data):
        q = row.get("question", "")
        a = row.get("answer", "")

        trajectory, target_epoch = parse_qa_sample(q)
        target_poi = parse_answer_poi(a)
        if not trajectory or target_poi is None:
            continue
        if target_epoch is None:
            continue

        query_record = visits_to_record(trajectory, p2i)
        samples.append(
            {
                "sample_id": i,
                "query_record": query_record,
                "target_poi": target_poi,
                "target_epoch": int(target_epoch),
            }
        )

    samples.sort(key=lambda x: x["target_epoch"])
    print(f"Parsed {len(samples)} valid QA samples from {path}")
    return samples


def attach_positives(samples, index, use_time_mask, max_pos=64, drop_zero=True):
    out = []
    for s in samples:
        if use_time_mask:
            counts = index.positives_before(s["target_poi"], s["target_epoch"])
        else:
            counts = index.positives_all(s["target_poi"])

        if not counts:
            if drop_zero:
                continue
            s2 = dict(s)
            s2["pos_pool_indices"] = []
            s2["pos_weights"] = []
            out.append(s2)
            continue

        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if max_pos is not None and len(items) > max_pos:
            head = items[: max_pos // 2]
            tail = items[max_pos // 2 :]
            np.random.shuffle(tail)
            items = head + tail[: (max_pos - len(head))]

        # 待考虑频次的权重
        idxs = [pid for pid, _ in items]
        w = np.array([math.log1p(c) for _, c in items], dtype=np.float32)
        w = w / (w.sum() + 1e-8)

        s2 = dict(s)
        s2["pos_pool_indices"] = idxs
        s2["pos_weights"] = w.tolist()
        out.append(s2)

    return out


class QATrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    max_l = max(x["query_record"]["l"] for x in batch)
    bpis = torch.stack([F.pad(x["query_record"]["pis"], (0, max_l - x["query_record"]["l"])) for x in batch])
    bhs = torch.stack([F.pad(x["query_record"]["hs"], (0, max_l - x["query_record"]["l"])) for x in batch])
    bds = torch.stack([F.pad(x["query_record"]["ds"], (0, max_l - x["query_record"]["l"])) for x in batch])
    bl = torch.tensor([x["query_record"]["l"] for x in batch], dtype=torch.long)

    return {
        "sample_id": [x["sample_id"] for x in batch],
        "query_pis": bpis,
        "query_hs": bhs,
        "query_ds": bds,
        "query_l": bl,
        "pos_pool_indices": [x["pos_pool_indices"] for x in batch],
        "pos_weights": [x["pos_weights"] for x in batch],
    }


def compute_multipos_loss(scores, pos_cols, pos_weights):
    log_probs = F.log_softmax(scores, dim=1)
    target = torch.zeros_like(scores)

    for i in range(scores.size(0)):
        idxs = pos_cols[i]
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
    return -(target[valid] * log_probs[valid]).sum(dim=1).mean()


def compute_rank_hinge(sim_scores, pos_cols, margin=0.05, hard_neg_k=8):
    bsz, n = sim_scores.shape
    losses = []
    for i in range(bsz):
        pos = pos_cols[i]
        if not pos:
            continue

        pos_t = torch.tensor(pos, dtype=torch.long, device=sim_scores.device)
        pos_s = sim_scores[i, pos_t].mean()

        neg_mask = torch.ones(n, dtype=torch.bool, device=sim_scores.device)
        neg_mask[pos_t] = False
        neg_scores = sim_scores[i].masked_fill(~neg_mask, float("-inf"))
        k = min(hard_neg_k, int(neg_mask.sum().item()))
        if k <= 0:
            continue

        hard_vals = torch.topk(neg_scores, k=k).values
        losses.append(F.relu(margin - pos_s + hard_vals).mean())

    if not losses:
        return sim_scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def evaluate_coverage(samples, model, pool_records, sem_all, geo_all, device, top_ks=(10, 20, 50, 100), count_topk=10):
    dev = torch.device(device)
    model.eval()

    with torch.no_grad():
        pool_emb = []
        for i in range(0, len(pool_records), 256):
            z = encode_records(pool_records[i : i + 256], model, sem_all, geo_all, dev, model.shared_dim)
            pool_emb.append(z)
        pool_emb = torch.cat(pool_emb, dim=0)

    hit = {k: 0 for k in top_ks}
    pos3 = 0
    pos5 = 0
    total = 0

    for s in samples:
        pos_set = set(s["pos_pool_indices"])
        if not pos_set:
            continue

        with torch.no_grad():
            zq = encode_records([s["query_record"]], model, sem_all, geo_all, dev, model.shared_dim)
            score = (zq @ pool_emb.t()).squeeze(0)

        max_k = min(max(top_ks), score.numel())
        top_ids = torch.topk(score, k=max_k).indices.tolist()

        for k in top_ks:
            ids = top_ids[:k]
            if any(i in pos_set for i in ids):
                hit[k] += 1

        ids10 = top_ids[:count_topk]
        c = sum(1 for i in ids10 if i in pos_set)
        if c >= 3:
            pos3 += 1
        if c >= 5:
            pos5 += 1
        total += 1

    if total == 0:
        out = {f"coverage@{k}": 0.0 for k in top_ks}
        out[f"pos>=3@{count_topk}"] = 0.0
        out[f"pos>=5@{count_topk}"] = 0.0
        return out

    out = {f"coverage@{k}": hit[k] / total for k in top_ks}
    out[f"pos>=3@{count_topk}"] = pos3 / total
    out[f"pos>=5@{count_topk}"] = pos5 / total
    return out


def plot_training_curves(history, save_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skip plotting")
        return

    if not history:
        return

    x = [h["epoch"] for h in history]
    y_total = [h["loss"] for h in history]
    y_main = [h["loss_main"] for h in history]
    y_rank_w = [h["loss_rank_weighted"] for h in history]
    y_rank_raw = [h["loss_rank_raw"] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_total, label="Total loss", linewidth=2.2)
    plt.plot(x, y_main, label="Main loss", linewidth=1.8)
    plt.plot(x, y_rank_w, label="Rank loss (weighted)", linewidth=1.8)
    plt.plot(x, y_rank_raw, label="Rank loss (raw)", linewidth=1.2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Trajectory Retriever Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def train_trajectory(
    dataset,
    train_qa,
    val_qa,
    save_dir="./traj_lib/",
    encoder_name="bert",
    epochs=120,
    batch_size=64,
    lr=3e-4,
    patience=12,
    tau=0.1,
    shared_dim=128,
    max_pos=64,
    num_neg=256,
    hard_neg_k=8,
    lambda_rank=0.1,
    rank_margin=0.05,
    device="cpu",
):
    os.makedirs(save_dir, exist_ok=True)
    dev = torch.device(device)

    all_pois = sorted(dataset.poi_dict.keys())
    p2i = {p: i for i, p in enumerate(all_pois)}
    n_pois = len(all_pois)

    print("Loading trajectory features...")
    try:
        from train_semantic_view import prepare_features

        fd = prepare_features(dataset, None, all_pois, encoder_name, device, True)
    except Exception:
        sem_v = np.random.randn(n_pois, 32).astype(np.float32)
        lats = np.array([dataset.poi_dict[p].latitude for p in all_pois])
        lons = np.array([dataset.poi_dict[p].longitude for p in all_pois])
        r = 6378137.0
        mx = r * np.radians(lons)
        my = r * np.log(np.tan(np.pi / 4 + np.radians(lats) / 2))
        mx_m, mx_s = mx.mean(), mx.std() + 1e-6
        my_m, my_s = my.mean(), my.std() + 1e-6
        geo_f = np.zeros((n_pois, 6), dtype=np.float32)
        for i in range(n_pois):
            geo_f[i] = [
                (mx[i] - mx_m) / mx_s,
                (my[i] - my_m) / my_s,
                np.sin(np.radians(lats[i])),
                np.cos(np.radians(lats[i])),
                np.sin(np.radians(lons[i])),
                np.cos(np.radians(lons[i])),
            ]
        fd = {"sem_vectors": sem_v, "geo_features": geo_f, "llm_dim": 32}

    sem_all = torch.tensor(fd["sem_vectors"], dtype=torch.float32, device=dev)
    geo_all = torch.tensor(fd["geo_features"], dtype=torch.float32, device=dev)
    llm_dim = int(fd["llm_dim"])
    geo_dim = int(fd["geo_features"].shape[1])

    pool_records, pool_meta, target_index, source_attr = build_pool(dataset, p2i)
    pool_size = len(pool_records)
    if pool_size == 0:
        raise RuntimeError("Empty trajectory pool")

    train_raw = build_samples_from_qa(train_qa, p2i)
    val_raw = build_samples_from_qa(val_qa, p2i)

    train_samples = attach_positives(train_raw, target_index, use_time_mask=True, max_pos=max_pos, drop_zero=True)
    val_samples = attach_positives(val_raw, target_index, use_time_mask=False, max_pos=None, drop_zero=False)

    if len(train_samples) == 0:
        raise RuntimeError("No train samples after attaching positives")
    if len(val_samples) == 0:
        raise RuntimeError("No val samples parsed")

    train_loader = torch.utils.data.DataLoader(
        QATrajectoryDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = TrajectoryEncoder(llm_dim=llm_dim, geo_dim=geo_dim, hidden=128, shared_dim=shared_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_metric = -1.0
    wait = 0
    history = []

    print(
        f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | "
        f"Trajectory pool: {pool_size} | Pos keys: {len(target_index.by_target)}"
    )

    for ep in range(1, epochs + 1):
        model.train()
        run_total = 0.0
        run_main = 0.0
        run_rank_raw = 0.0
        run_rank_w = 0.0
        n_batch = 0

        for batch in train_loader:
            opt.zero_grad()

            qpis = batch["query_pis"].to(dev)
            qhs = batch["query_hs"].to(dev)
            qds = batch["query_ds"].to(dev)
            ql = batch["query_l"].to(dev)

            z_q = model(sem_all[qpis], geo_all[qpis], qhs, qds, ql)

            pos_union = set()
            for idxs in batch["pos_pool_indices"]:
                pos_union.update(idxs)

            neg_candidates = [i for i in range(pool_size) if i not in pos_union]
            if len(neg_candidates) > num_neg:
                neg_idx = np.random.choice(neg_candidates, size=num_neg, replace=False).tolist()
            else:
                neg_idx = list(neg_candidates)

            cand_pool_indices = list(pos_union) + neg_idx
            if not cand_pool_indices:
                continue

            cand_records = [pool_records[i] for i in cand_pool_indices]
            z_cand = encode_records(cand_records, model, sem_all, geo_all, dev, shared_dim)

            col_map = {pool_idx: col for col, pool_idx in enumerate(cand_pool_indices)}
            pos_cols = []
            pos_w = []
            for i in range(len(batch["pos_pool_indices"])):
                cols = []
                ws = []
                for pool_idx, w in zip(batch["pos_pool_indices"][i], batch["pos_weights"][i]):
                    col = col_map.get(pool_idx)
                    if col is not None:
                        cols.append(col)
                        ws.append(float(w))
                if ws:
                    s = sum(ws)
                    ws = [x / (s + 1e-8) for x in ws]
                pos_cols.append(cols)
                pos_w.append(ws)

            sim_scores = z_q @ z_cand.t()
            scores = sim_scores / tau

            loss_main = compute_multipos_loss(scores, pos_cols, pos_w)
            if loss_main is None:
                continue
            loss_rank_raw = compute_rank_hinge(sim_scores, pos_cols, margin=rank_margin, hard_neg_k=hard_neg_k)
            loss_rank = lambda_rank * loss_rank_raw
            loss = loss_main + loss_rank

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_total += loss.item()
            run_main += loss_main.item()
            run_rank_raw += loss_rank_raw.item()
            run_rank_w += loss_rank.item()
            n_batch += 1

        sched.step()
        avg_total = run_total / max(1, n_batch)
        avg_main = run_main / max(1, n_batch)
        avg_rank_raw = run_rank_raw / max(1, n_batch)
        avg_rank_w = run_rank_w / max(1, n_batch)

        val_cov = evaluate_coverage(
            val_samples,
            model,
            pool_records,
            sem_all,
            geo_all,
            device=device,
            top_ks=(1, 5, 10, 20, 50),
            count_topk=10,
        )
        key_metric = val_cov["coverage@20"]

        history.append(
            {
                "epoch": ep,
                "loss": avg_total,
                "loss_main": avg_main,
                "loss_rank_raw": avg_rank_raw,
                "loss_rank_weighted": avg_rank_w,
                **val_cov,
            }
        )

        print(
            f"E{ep:03d} | loss={avg_total:.4f} "
            f"(main={avg_main:.4f}, rank_w={avg_rank_w:.4f}, rank_raw={avg_rank_raw:.4f}) "
            f"| cov@10={val_cov['coverage@10']:.4f} | cov@20={val_cov['coverage@20']:.4f} "
            f"| cov@50={val_cov['coverage@50']:.4f} | pos>=3@10={val_cov['pos>=3@10']:.4f} "
            f"| pos>=5@10={val_cov['pos>=5@10']:.4f}"
        )

        if key_metric > best_metric + 1e-4:
            best_metric = key_metric
            wait = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "traj_encoder.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    model.load_state_dict(torch.load(os.path.join(save_dir, "traj_encoder.pth"), map_location=dev))
    model.eval()

    with torch.no_grad():
        emb_chunks = []
        for i in range(0, len(pool_records), 256):
            z = encode_records(pool_records[i : i + 256], model, sem_all, geo_all, dev, shared_dim)
            emb_chunks.append(z.cpu())
        traj_pool_emb = torch.cat(emb_chunks, dim=0).numpy()

    np.save(os.path.join(save_dir, "traj_pool_embeddings.npy"), traj_pool_emb)
    with open(os.path.join(save_dir, "traj_pool_meta.json"), "w", encoding="utf-8") as f:
        json.dump(pool_meta, f, ensure_ascii=False)
    with open(os.path.join(save_dir, "poi_id_list.json"), "w", encoding="utf-8") as f:
        json.dump(all_pois, f, ensure_ascii=False)

    torch.save(
        {
            "shared_dim": shared_dim,
            "llm_dim": llm_dim,
            "geo_dim": geo_dim,
            "num_pois": n_pois,
            "tau": tau,
            "num_neg": num_neg,
            "max_pos": max_pos,
            "hard_neg_k": hard_neg_k,
            "lambda_rank": lambda_rank,
            "rank_margin": rank_margin,
            "best_val_coverage20": best_metric,
            "pool_source": source_attr,
            "train_qa": train_qa,
            "val_qa": val_qa,
            "val_positive_rule": "full_time_no_mask",
        },
        os.path.join(save_dir, "config.pth"),
    )

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
    p.add_argument("--val_qa", default="./datasets/nyc/preprocessed/validate_qa_pairs_kqt.json")
    p.add_argument("--save_dir", default="./traj_lib/")
    p.add_argument("--encoder", default="bert")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--shared_dim", type=int, default=128)
    p.add_argument("--max_pos", type=int, default=64)
    p.add_argument("--num_neg", type=int, default=256)
    p.add_argument("--hard_neg_k", type=int, default=8)
    p.add_argument("--lambda_rank", type=float, default=0.1)
    p.add_argument("--rank_margin", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ds = load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.val_qa)

    train_trajectory(
        dataset=ds,
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
        max_pos=args.max_pos,
        num_neg=args.num_neg,
        hard_neg_k=args.hard_neg_k,
        lambda_rank=args.lambda_rank,
        rank_margin=args.rank_margin,
        device=args.device,
    )


if __name__ == "__main__":
    main()
