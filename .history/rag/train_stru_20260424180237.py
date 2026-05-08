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
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

from poi_data_loader import load_full_dataset
from transition_graph import TimeBucketManager


class StructuralEncoder(nn.Module):
    def __init__(self, node_feat_dim=4, hidden=64, shared_dim=128, num_buckets=8, time_emb_dim=16, heads=4):
        super().__init__()
        self.shared_dim = shared_dim
        self.time_emb = nn.Embedding(num_buckets, time_emb_dim)
        self.input_proj = nn.Linear(node_feat_dim + time_emb_dim, hidden)
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=0.1)
        self.out_proj = nn.Linear(hidden, shared_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        t = self.time_emb(data.time_bucket)[batch]
        h = F.elu(self.input_proj(torch.cat([x, t], dim=-1)))
        h = F.elu(self.gat1(h, edge_index))
        h = self.gat2(h, edge_index)
        return F.normalize(self.out_proj(global_mean_pool(h, batch)), dim=-1)


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
        disable_cudnn_rnn=False,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.hour_emb = nn.Embedding(24, hour_emb_dim)
        self.dow_emb = nn.Embedding(7, dow_emb_dim)
        tdim = hour_emb_dim + dow_emb_dim

        self.poi_proj = nn.Sequential(nn.Linear(llm_dim, hidden), nn.ReLU())
        self.geo_proj = nn.Sequential(nn.Linear(geo_dim, hidden), nn.ReLU())
        self.time_proj = nn.Sequential(nn.Linear(tdim, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden * 3, hidden, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        # Some CUDA environments fail during GRU flatten with cuDNN init errors.
        # Allow a safe fallback path that skips cuDNN RNN weight flattening.
        if disable_cudnn_rnn:
            self.gru.flatten_parameters = lambda: None
        self.out_proj = nn.Sequential(nn.Linear(hidden, shared_dim), nn.LayerNorm(shared_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, llm_seq, geo_seq, hours, dows, lengths):
        h_poi = self.poi_proj(llm_seq)
        h_geo = self.geo_proj(geo_seq)
        h_time = self.time_proj(torch.cat([self.hour_emb(hours), self.dow_emb(dows)], dim=-1))
        x = self.dropout(torch.cat([h_poi, h_geo, h_time], dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return F.normalize(self.out_proj(h_n[-1]), dim=-1)


class StructQueryProjector(nn.Module):
    def __init__(self, shared_dim=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(shared_dim, shared_dim), nn.LayerNorm(shared_dim), nn.Dropout(0.1))

    def forward(self, traj_emb):
        return F.normalize(self.proj(traj_emb), dim=-1)


class TemporalEdgeIndex:
    def __init__(self):
        self.edge_times = defaultdict(list)
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        self.out_totals = defaultdict(list)
        self.in_totals = defaultdict(list)

    def add(self, epoch, src, dst, bucket):
        self.edge_times[(bucket, src, dst)].append(epoch)
        self.outgoing[(bucket, src)].append((epoch, dst))
        self.incoming[(bucket, dst)].append((epoch, src))
        self.out_totals[(bucket, src)].append(epoch)
        self.in_totals[(bucket, dst)].append(epoch)

    def build(self):
        for k in self.edge_times:
            self.edge_times[k].sort()
        for k in self.outgoing:
            self.outgoing[k].sort()
        for k in self.incoming:
            self.incoming[k].sort()
        for k in self.out_totals:
            self.out_totals[k].sort()
        for k in self.in_totals:
            self.in_totals[k].sort()

    def edge_count_before(self, bucket, src, dst, cutoff):
        arr = self.edge_times.get((bucket, src, dst), [])
        return bisect.bisect_left(arr, cutoff)

    def edge_count_all(self, bucket, src, dst):
        return len(self.edge_times.get((bucket, src, dst), []))

    def neighbors_before(self, src, bucket, cutoff):
        entries = self.outgoing.get((bucket, src), [])
        if not entries:
            return {}
        n = bisect.bisect_left([e[0] for e in entries], cutoff)
        cnt = defaultdict(int)
        for i in range(n):
            cnt[entries[i][1]] += 1
        return dict(cnt)

    def neighbors_all(self, src, bucket):
        entries = self.outgoing.get((bucket, src), [])
        cnt = defaultdict(int)
        for _, d in entries:
            cnt[d] += 1
        return dict(cnt)

    def node_out_before(self, bucket, node, cutoff):
        arr = self.out_totals.get((bucket, node), [])
        return bisect.bisect_left(arr, cutoff)

    def node_out_all(self, bucket, node):
        return len(self.out_totals.get((bucket, node), []))

    def node_in_before(self, bucket, node, cutoff):
        arr = self.in_totals.get((bucket, node), [])
        return bisect.bisect_left(arr, cutoff)

    def node_in_all(self, bucket, node):
        return len(self.in_totals.get((bucket, node), []))

    def top_centers(self, bucket, top_k=64):
        rows = []
        for (b, src), arr in self.out_totals.items():
            if b != bucket:
                continue
            rows.append((src, len(arr)))
        rows.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in rows[:top_k]]


class SubgraphExtractor:
    def __init__(self, edge_index, num_buckets=8, max_nodes=20, top_k_centers=64):
        self.ei = edge_index
        self.num_buckets = num_buckets
        self.max_nodes = max_nodes
        self.node_feat_dim = 4
        self.centers_by_bucket = {}
        for b in range(num_buckets):
            self.centers_by_bucket[b] = edge_index.top_centers(b, top_k=top_k_centers)

    def _select_nodes(self, center, bucket, cutoff=None):
        nb = self.ei.neighbors_before(center, bucket, cutoff) if cutoff is not None else self.ei.neighbors_all(center, bucket)
        rows = sorted(nb.items(), key=lambda x: x[1], reverse=True)
        sel = [center]
        for nid, _ in rows[: self.max_nodes - 1]:
            if nid != center:
                sel.append(nid)
        if len(sel) < min(5, self.max_nodes):
            for x in self.centers_by_bucket.get(bucket, []):
                if x not in sel:
                    sel.append(x)
                if len(sel) >= min(5, self.max_nodes):
                    break
        return sel

    def _build_node_features(self, nodes, bucket, cutoff=None):
        feat = np.zeros((len(nodes), self.node_feat_dim), dtype=np.float32)
        for i, nid in enumerate(nodes):
            if cutoff is None:
                o = self.ei.node_out_all(bucket, nid)
                inn = self.ei.node_in_all(bucket, nid)
            else:
                o = self.ei.node_out_before(bucket, nid, cutoff)
                inn = self.ei.node_in_before(bucket, nid, cutoff)
            nb = self.ei.neighbors_before(nid, bucket, cutoff) if cutoff is not None else self.ei.neighbors_all(nid, bucket)
            feat[i, 0] = float(o)
            feat[i, 1] = float(inn)
            feat[i, 2] = float(len(nb))
            feat[i, 3] = float(sum(nb.values()))
        scale = feat.max(axis=0, keepdims=True) + 1e-6
        return feat / scale

    def extract(self, center, bucket, cutoff=None):
        nodes = self._select_nodes(center, bucket, cutoff=cutoff)
        nmap = {nid: i for i, nid in enumerate(nodes)}
        srcs, dsts, ews = [], [], []

        for ni in nodes:
            for nj in nodes:
                if ni == nj:
                    continue
                w = self.ei.edge_count_before(bucket, ni, nj, cutoff) if cutoff is not None else self.ei.edge_count_all(bucket, ni, nj)
                if w > 0:
                    srcs.append(nmap[ni])
                    dsts.append(nmap[nj])
                    ews.append(float(w))

        if not srcs:
            srcs, dsts, ews = [0], [0], [1.0]

        x = torch.tensor(self._build_node_features(nodes, bucket, cutoff=cutoff), dtype=torch.float32)
        return Data(
            x=x,
            edge_index=torch.tensor([srcs, dsts], dtype=torch.long),
            edge_attr=torch.tensor(ews, dtype=torch.float32).unsqueeze(1),
            original_ids=torch.tensor(nodes, dtype=torch.long),
            time_bucket=torch.tensor([bucket], dtype=torch.long),
            num_nodes=len(nodes),
        )


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
        traj_text = m2.group(1) if m2 else question_text.split("Given the data,")[0]

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
                    "hour": dt.hour,
                    "day_of_week": dt.weekday(),
                    "category": cat_name.strip(),
                    "category_id": int(cat_id),
                }
            )
        except Exception:
            continue

    tp = r"Given the data,\s+At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+Which"
    tm = re.search(tp, question_text)
    if tm:
        try:
            target_epoch = int(datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S").timestamp())
        except Exception:
            target_epoch = None
    return trajectory, target_epoch


def parse_answer_poi(answer_text):
    m = re.search(r"POI\s+id\s+(\d+)", answer_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b", answer_text)
    return int(m.group(1)) if m else None


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
            print(f"Using dataset.{attr} as training graph source")
            return val, attr
    print("dataset.train_trajectories not found; fallback to all_trajectories")
    return dataset.all_trajectories, "all_trajectories(fallback)"


def visits_to_record(visits, p2i):
    pis = [p2i.get(int(v["poi_id"]), 0) for v in visits]
    hs = [int(v.get("hour", 0)) % 24 for v in visits]
    ds = [int(v.get("day_of_week", 0)) % 7 for v in visits]
    if not pis:
        pis, hs, ds = [0], [0], [0]
    return {"pis": torch.tensor(pis, dtype=torch.long), "hs": torch.tensor(hs, dtype=torch.long), "ds": torch.tensor(ds, dtype=torch.long), "l": len(pis)}


def encode_traj_records(records, model, sem_all, geo_all, dev):
    # Be robust to mixed inputs: raw variable-length records and pre-padded rows.
    # We always trust `l` as the effective length and slice first, then re-pad.
    eff_l = []
    for r in records:
        l = int(r["l"])
        cur = min(l, int(r["pis"].numel()), int(r["hs"].numel()), int(r["ds"].numel()))
        eff_l.append(max(cur, 1))

    max_l = max(eff_l)

    def _stack_key(key):
        rows = []
        for r, l in zip(records, eff_l):
            t = r[key].reshape(-1)[:l]
            rows.append(F.pad(t, (0, max_l - l)))
        return torch.stack(rows)

    bp = _stack_key("pis").to(dev)
    bh = _stack_key("hs").to(dev)
    bd = _stack_key("ds").to(dev)
    bl = torch.tensor(eff_l, dtype=torch.long, device=dev)
    return model(sem_all[bp], geo_all[bp], bh, bd, bl)


def _bucket_from_last_visit(tm, visit):
    if hasattr(tm, "get_bucket"):
        return int(tm.get_bucket(int(visit["hour"]), int(visit["day_of_week"]) >= 5))
    return int(visit["hour"] // 3)


def build_samples_from_qa(path, tm, p2i):
    qa_data = load_qa_data(path)
    samples = []
    for i, row in enumerate(qa_data):
        q = row.get("question", "")
        a = row.get("answer", "")
        traj, target_epoch = parse_qa_sample(q)
        target_poi = parse_answer_poi(a)
        if not traj or target_poi is None or target_epoch is None:
            continue
        bucket = _bucket_from_last_visit(tm, traj[-1])
        samples.append(
            {
                "sample_id": i,
                "query_record": visits_to_record(traj, p2i),
                "target_poi": target_poi,
                "target_epoch": int(target_epoch),
                "bucket": bucket,
            }
        )
    samples.sort(key=lambda x: x["target_epoch"])
    print(f"Parsed {len(samples)} valid QA samples from {path}")
    return samples


def build_training_graph(dataset, p2i, tm):
    trajs, source_name = _get_train_trajectories(dataset)
    ei = TemporalEdgeIndex()
    for traj in trajs.values():
        visits = getattr(traj, "visits", [])
        for i in range(len(visits) - 1):
            s, d = visits[i], visits[i + 1]
            si, di = p2i.get(int(s.poi_id), -1), p2i.get(int(d.poi_id), -1)
            if si < 0 or di < 0:
                continue
            b = int(tm.get_bucket_from_visit(s))
            ei.add(int(s.epoch), si, di, b)
    ei.build()
    return ei, source_name


def build_subgraph_pool(extractor):
    pool_meta = []
    for b in range(extractor.num_buckets):
        for c in extractor.centers_by_bucket.get(b, []):
            sg = extractor.extract(c, b, cutoff=None)
            nodes = set(sg.original_ids.tolist())
            pool_meta.append({"center": int(c), "bucket": int(b), "nodes": sorted(int(x) for x in nodes)})
    return pool_meta


def attach_positives(samples, pool_meta, p2i, ei, use_time_mask=True, max_pos=64, drop_zero=True):
    by_bucket = defaultdict(list)
    for i, m in enumerate(pool_meta):
        by_bucket[int(m["bucket"])].append((i, int(m["center"])))

    out = []
    for s in samples:
        tgt = p2i.get(int(s["target_poi"]), -1)
        if tgt < 0:
            continue

        pos = []
        for sg_idx, center in by_bucket.get(int(s["bucket"]), []):
            if use_time_mask:
                nb = ei.neighbors_before(center, int(s["bucket"]), int(s["target_epoch"]))
            else:
                nb = ei.neighbors_all(center, int(s["bucket"]))
            if tgt in nb:
                pos.append((sg_idx, nb[tgt]))

        if not pos:
            if drop_zero:
                continue
            s2 = dict(s)
            s2["pos_sg_indices"] = []
            s2["pos_weights"] = []
            out.append(s2)
            continue

        pos = sorted(pos, key=lambda x: x[1], reverse=True)
        if max_pos is not None and len(pos) > max_pos:
            head = pos[: max_pos // 2]
            tail = pos[max_pos // 2 :]
            np.random.shuffle(tail)
            pos = head + tail[: (max_pos - len(head))]

        idxs = [x[0] for x in pos]
        w = np.array([math.log1p(x[1]) for x in pos], dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        s2 = dict(s)
        s2["pos_sg_indices"] = idxs
        s2["pos_weights"] = w.tolist()
        out.append(s2)
    return out


class QAStructDataset(torch.utils.data.Dataset):
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
        "query_pis": bpis,
        "query_hs": bhs,
        "query_ds": bds,
        "query_l": bl,
        "bucket": torch.tensor([x["bucket"] for x in batch], dtype=torch.long),
        "target_epoch": [x["target_epoch"] for x in batch],
        "pos_sg_indices": [x["pos_sg_indices"] for x in batch],
        "pos_weights": [x["pos_weights"] for x in batch],
    }


def compute_multipos_loss(scores, pos_cols, pos_weights):
    log_probs = F.log_softmax(scores, dim=1)
    target = torch.zeros_like(scores)
    for i in range(scores.size(0)):
        if not pos_cols[i]:
            continue
        idx_t = torch.tensor(pos_cols[i], dtype=torch.long, device=scores.device)
        w_t = torch.tensor(pos_weights[i], dtype=torch.float32, device=scores.device)
        target[i, idx_t] = w_t
    valid = target.sum(dim=1) > 0
    if valid.sum().item() == 0:
        return None
    return -(target[valid] * log_probs[valid]).sum(dim=1).mean()


def compute_rank_hinge(sim_scores, pos_cols, allowed_cols=None, margin=0.05, hard_neg_k=8):
    bsz, n = sim_scores.shape
    losses = []
    for i in range(bsz):
        pos = pos_cols[i]
        if not pos:
            continue
        pos_t = torch.tensor(pos, dtype=torch.long, device=sim_scores.device)
        pos_s = sim_scores[i, pos_t].mean()
        if allowed_cols is None:
            mask = torch.ones(n, dtype=torch.bool, device=sim_scores.device)
        else:
            mask = torch.zeros(n, dtype=torch.bool, device=sim_scores.device)
            if allowed_cols[i]:
                allow_t = torch.tensor(allowed_cols[i], dtype=torch.long, device=sim_scores.device)
                mask[allow_t] = True
        mask[pos_t] = False
        neg = sim_scores[i].masked_fill(~mask, float("-inf"))
        k = min(hard_neg_k, int(mask.sum().item()))
        if k <= 0:
            continue
        hard = torch.topk(neg, k=k).values
        losses.append(F.relu(margin - pos_s + hard).mean())
    if not losses:
        return sim_scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def encode_subgraphs(indices, pool_meta, extractor, str_enc, dev, cutoff=None):
    rows = []
    for idx in indices:
        m = pool_meta[idx]
        sg = extractor.extract(m["center"], m["bucket"], cutoff=cutoff)
        sg.batch = torch.zeros(sg.num_nodes, dtype=torch.long)
        rows.append(str_enc(sg.to(dev)).squeeze(0))
    return torch.stack(rows, dim=0)


def evaluate_coverage(val_samples, pool_meta, extractor, traj_enc, struct_proj, str_enc, sem_all, geo_all, dev, top_ks=(10, 20, 50), count_topk=10):
    traj_enc.eval()
    struct_proj.eval()
    str_enc.eval()
    with torch.no_grad():
        all_idx = list(range(len(pool_meta)))
        pool_emb = encode_subgraphs(all_idx, pool_meta, extractor, str_enc, dev, cutoff=None)

    hit = {k: 0 for k in top_ks}
    pos3, pos5, total = 0, 0, 0
    for s in val_samples:
        pos_set = set(s["pos_sg_indices"])
        if not pos_set:
            continue
        with torch.no_grad():
            zt = encode_traj_records([s["query_record"]], traj_enc, sem_all, geo_all, dev)
            zq = struct_proj(zt)
            score = (zq @ pool_emb.t()).squeeze(0)
        max_k = min(max(top_ks), score.numel())
        ids = torch.topk(score, k=max_k).indices.tolist()
        for k in top_ks:
            if any(i in pos_set for i in ids[:k]):
                hit[k] += 1
        c = sum(1 for i in ids[:count_topk] if i in pos_set)
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
        return
    x = [h["epoch"] for h in history]
    plt.figure(figsize=(10, 6))
    plt.plot(x, [h["loss"] for h in history], label="Total loss", linewidth=2.2)
    plt.plot(x, [h["loss_main"] for h in history], label="Main loss", linewidth=1.8)
    plt.plot(x, [h["loss_rank_weighted"] for h in history], label="Rank loss (weighted)", linewidth=1.8)
    plt.plot(x, [h["loss_rank_raw"] for h in history], label="Rank loss (raw)", linewidth=1.2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Structural Retriever Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def train_structural(
    dataset,
    train_qa,
    val_qa,
    save_dir="./struct_lib/",
    encoder_name="bert",
    epochs=120,
    batch_size=32,
    lr=3e-4,
    patience=12,
    tau=0.1,
    shared_dim=128,
    max_pos=96,
    num_neg_per_query=64,
    hard_neg_k=8,
    lambda_rank=0.1,
    rank_margin=0.05,
    top_k_centers=128,
    max_nodes=32,
    disable_cudnn_rnn=False,
    device="cpu",
):
    os.makedirs(save_dir, exist_ok=True)
    dev = torch.device(device)

    tm = TimeBucketManager()
    all_pois = sorted(dataset.poi_dict.keys())
    p2i = {p: i for i, p in enumerate(all_pois)}
    n_pois = len(all_pois)

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

    ei, pool_source = build_training_graph(dataset, p2i, tm)
    extractor = SubgraphExtractor(ei, num_buckets=getattr(tm, "num_buckets", 8), max_nodes=max_nodes, top_k_centers=top_k_centers)
    pool_meta = build_subgraph_pool(extractor)
    if not pool_meta:
        raise RuntimeError("Subgraph pool is empty")

    train_raw = build_samples_from_qa(train_qa, tm, p2i)
    val_raw = build_samples_from_qa(val_qa, tm, p2i)
    train_samples = attach_positives(train_raw, pool_meta, p2i, ei, use_time_mask=True, max_pos=max_pos, drop_zero=True)
    val_samples = attach_positives(val_raw, pool_meta, p2i, ei, use_time_mask=False, max_pos=None, drop_zero=False)
    if not train_samples:
        raise RuntimeError("No train samples after positive attach")

    loader = torch.utils.data.DataLoader(
        QAStructDataset(train_samples), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn
    )

    try:
        traj_enc = TrajectoryEncoder(
            llm_dim=llm_dim,
            geo_dim=geo_dim,
            hidden=128,
            shared_dim=shared_dim,
            disable_cudnn_rnn=disable_cudnn_rnn,
        ).to(dev)
    except RuntimeError as e:
        msg = str(e)
        if dev.type == "cuda" and "CUDNN_STATUS_NOT_INITIALIZED" in msg:
            print("[Warn] cuDNN RNN init failed. Retrying with cuDNN disabled for GRU.")
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.cuda.empty_cache()
            traj_enc = TrajectoryEncoder(
                llm_dim=llm_dim,
                geo_dim=geo_dim,
                hidden=128,
                shared_dim=shared_dim,
                disable_cudnn_rnn=True,
            ).to(dev)
        else:
            raise
    struct_proj = StructQueryProjector(shared_dim=shared_dim).to(dev)
    str_enc = StructuralEncoder(node_feat_dim=extractor.node_feat_dim, hidden=64, shared_dim=shared_dim, num_buckets=extractor.num_buckets).to(dev)
    params = list(traj_enc.parameters()) + list(struct_proj.parameters()) + list(str_enc.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_metric = -1.0
    wait = 0
    history = []
    pool_size = len(pool_meta)
    bucket_to_pool = defaultdict(list)
    for i, meta in enumerate(pool_meta):
        bucket_to_pool[int(meta["bucket"])].append(i)

    print(f"Train samples={len(train_samples)} | Val samples={len(val_samples)} | SG pool={pool_size}")
    for ep in range(1, epochs + 1):
        traj_enc.train()
        struct_proj.train()
        str_enc.train()
        run_total = run_main = run_rank_raw = run_rank_w = 0.0
        n_batch = 0

        for batch in loader:
            opt.zero_grad()
            # Batch is already padded by collate_fn; feed directly to avoid re-padding mismatch.
            bp = batch["query_pis"].to(dev)
            bh = batch["query_hs"].to(dev)
            bd = batch["query_ds"].to(dev)
            bl = batch["query_l"].to(dev)
            z_traj = traj_enc(sem_all[bp], geo_all[bp], bh, bd, bl)
            z_q = struct_proj(z_traj)

            allowed_cols = []
            cand_indices_set = set()

            for i in range(len(batch["pos_sg_indices"])):
                pos_set = set(batch["pos_sg_indices"][i])
                b_i = int(batch["bucket"][i].item())
                same_bucket = bucket_to_pool.get(b_i, [])
                neg_candidates = [x for x in same_bucket if x not in pos_set]
                if len(neg_candidates) > num_neg_per_query:
                    neg_sel = np.random.choice(
                        neg_candidates, size=num_neg_per_query, replace=False
                    ).tolist()
                else:
                    neg_sel = neg_candidates

                allowed_global = list(pos_set) + neg_sel
                allowed_cols.append(allowed_global)
                cand_indices_set.update(allowed_global)

            cand_indices = sorted(cand_indices_set)
            if not cand_indices:
                continue

            # Encode candidate subgraphs with full edges. Time masking is only used when building positives.
            z_cand = encode_subgraphs(cand_indices, pool_meta, extractor, str_enc, dev, cutoff=None)

            col_map = {pi: c for c, pi in enumerate(cand_indices)}
            mapped_pos_cols, mapped_pos_w = [], []
            mapped_allowed_cols = []
            for i in range(len(allowed_cols)):
                cols, ws, allow = [], [], []
                for pi in allowed_cols[i]:
                    c = col_map.get(pi)
                    if c is not None:
                        allow.append(c)
                for pi, w in zip(batch["pos_sg_indices"][i], batch["pos_weights"][i]):
                    c = col_map.get(pi)
                    if c is not None:
                        cols.append(c)
                        ws.append(float(w))
                if ws:
                    s = sum(ws)
                    ws = [x / (s + 1e-8) for x in ws]
                mapped_pos_cols.append(cols)
                mapped_pos_w.append(ws)
                mapped_allowed_cols.append(allow)

            sim_scores = z_q @ z_cand.t()
            scores = sim_scores / tau
            allowed_mask = torch.zeros_like(scores, dtype=torch.bool)
            for i, allow in enumerate(mapped_allowed_cols):
                if allow:
                    allow_t = torch.tensor(allow, dtype=torch.long, device=dev)
                    allowed_mask[i, allow_t] = True
            scores = scores.masked_fill(~allowed_mask, -1e9)

            loss_main = compute_multipos_loss(scores, mapped_pos_cols, mapped_pos_w)
            if loss_main is None:
                continue
            loss_rank_raw = compute_rank_hinge(
                sim_scores,
                mapped_pos_cols,
                allowed_cols=mapped_allowed_cols,
                margin=rank_margin,
                hard_neg_k=hard_neg_k,
            )
            loss_rank = lambda_rank * loss_rank_raw
            loss = loss_main + loss_rank

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
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

        val_cov = evaluate_coverage(val_samples, pool_meta, extractor, traj_enc, struct_proj, str_enc, sem_all, geo_all, dev, top_ks=(1, 5, 10, 20, 50), count_topk=10)
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
            f"E{ep:03d} | loss={avg_total:.4f} (main={avg_main:.4f}, rank_w={avg_rank_w:.4f}, rank_raw={avg_rank_raw:.4f}) "
            f"| cov@10={val_cov['coverage@10']:.4f} | cov@20={val_cov['coverage@20']:.4f} | cov@50={val_cov['coverage@50']:.4f} "
            f"| pos>=3@10={val_cov['pos>=3@10']:.4f} | pos>=5@10={val_cov['pos>=5@10']:.4f}"
        )

        if key_metric > best_metric + 1e-4:
            best_metric = key_metric
            wait = 0
            torch.save(
                {
                    "traj_enc": traj_enc.state_dict(),
                    "struct_proj": struct_proj.state_dict(),
                    "str_enc": str_enc.state_dict(),
                },
                os.path.join(save_dir, "struct_model.pth"),
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    ckpt = torch.load(os.path.join(save_dir, "struct_model.pth"), map_location=dev)
    traj_enc.load_state_dict(ckpt["traj_enc"])
    struct_proj.load_state_dict(ckpt["struct_proj"])
    str_enc.load_state_dict(ckpt["str_enc"])
    str_enc.eval()

    with torch.no_grad():
        all_idx = list(range(len(pool_meta)))
        sg_pool_emb = encode_subgraphs(all_idx, pool_meta, extractor, str_enc, dev, cutoff=None).cpu().numpy()

    np.save(os.path.join(save_dir, "sg_pool_embeddings.npy"), sg_pool_emb)
    with open(os.path.join(save_dir, "sg_pool_meta.json"), "w", encoding="utf-8") as f:
        json.dump(pool_meta, f, ensure_ascii=False)
    with open(os.path.join(save_dir, "poi_id_list.json"), "w", encoding="utf-8") as f:
        json.dump(all_pois, f, ensure_ascii=False)

    torch.save(
        {
            "shared_dim": shared_dim,
            "llm_dim": llm_dim,
            "geo_dim": geo_dim,
            "node_feat_dim": extractor.node_feat_dim,
            "num_pois": n_pois,
            "num_buckets": extractor.num_buckets,
            "tau": tau,
            "num_neg_per_query": num_neg_per_query,
            "max_pos": max_pos,
            "hard_neg_k": hard_neg_k,
            "lambda_rank": lambda_rank,
            "rank_margin": rank_margin,
            "max_nodes": max_nodes,
            "top_k_centers": top_k_centers,
            "disable_cudnn_rnn": disable_cudnn_rnn,
            "best_val_coverage20": best_metric,
            "pool_source": pool_source,
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
    print(f"Saved loss curve to {curve_path}")
    print(f"Training done. Best coverage@20={best_metric:.4f}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="./datasets/nyc/preprocessed/train_sample.csv")
    p.add_argument("--test_csv", default="./datasets/nyc/preprocessed/test_sample.csv")
    p.add_argument("--train_qa", default="./datasets/nyc/preprocessed/train_qa_pairs_kqt.json")
    p.add_argument("--val_qa", default="./datasets/nyc/preprocessed/validate_qa_pairs_kqt.json")
    p.add_argument("--save_dir", default="./struct_lib/")
    p.add_argument("--encoder", default="bert")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--shared_dim", type=int, default=128)
    p.add_argument("--max_pos", type=int, default=96)
    p.add_argument("--num_neg_per_query", type=int, default=64)
    p.add_argument("--num_neg", type=int, default=None, help="Deprecated alias of --num_neg_per_query")
    p.add_argument("--hard_neg_k", type=int, default=8)
    p.add_argument("--lambda_rank", type=float, default=0.1)
    p.add_argument("--rank_margin", type=float, default=0.05)
    p.add_argument("--top_k_centers", type=int, default=128)
    p.add_argument("--max_nodes", type=int, default=32)
    p.add_argument("--disable_cudnn_rnn", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.num_neg is not None:
        args.num_neg_per_query = args.num_neg

    ds = load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.val_qa)
    train_structural(
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
        num_neg_per_query=args.num_neg_per_query,
        hard_neg_k=args.hard_neg_k,
        lambda_rank=args.lambda_rank,
        rank_margin=args.rank_margin,
        top_k_centers=args.top_k_centers,
        max_nodes=args.max_nodes,
        disable_cudnn_rnn=args.disable_cudnn_rnn,
        device=args.device,
    )


if __name__ == "__main__":
    main()
