import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError as exc:
    raise ImportError("torch_geometric is required for the structure view GAT encoder.") from exc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poi_data_loader import load_full_dataset  # noqa: E402
from transition_graph import TimeBucketManager  # noqa: E402


def load_json_or_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "<question>:" in line and "<answer>:" in line:
            q_part, a_part = line.split("<answer>:", 1)
            rows.append({"question": q_part.replace("<question>:", "").strip(), "answer": a_part.strip()})
        else:
            rows.append(json.loads(line))
    return rows


def parse_qa_sample(question: str) -> Tuple[List[dict], Optional[int], Optional[int], Optional[int]]:
    traj = []
    section_patterns = [
        r"\[Current trajectory'?s? check-in sequence\]:\s*(.*?)\s*\[Historical check-in sequences\]:",
        r"\[Current trajectory'?s? check-in sequence\]:\s*(.*?)\s*Given the data,",
    ]
    traj_text = None
    for pat in section_patterns:
        m = re.search(pat, question, re.DOTALL)
        if m:
            traj_text = m.group(1)
            break
    if traj_text is None:
        traj_text = question.split("Given the data,")[0]

    visit_pat = (
        r"At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+"
        r"user\s+\d+\s+visited\s+POI\s+id\s+(\d+)\s+"
        r"which\s+is\s+a\s+(.+?)\s+with\s+Category\s+id\s+(\d+)"
    )
    for m in re.finditer(visit_pat, traj_text):
        ts, poi_id, cat_name, cat_id = m.groups()
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        traj.append(
            {
                "poi_id": int(poi_id),
                "category": cat_name.strip(),
                "category_id": int(cat_id),
                "epoch": int(dt.timestamp()),
                "hour": int(dt.hour),
                "dow": int(dt.weekday()),
            }
        )

    target_epoch = None
    target_hour = None
    target_dow = None
    target_pat = r"Given the data,\s+At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+Which"
    tm = re.search(target_pat, question)
    if tm:
        try:
            dt = datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S")
            target_epoch = int(dt.timestamp())
            target_hour = int(dt.hour)
            target_dow = int(dt.weekday())
        except ValueError:
            pass
    return traj, target_epoch, target_hour, target_dow


def parse_answer_poi(answer: str) -> Optional[int]:
    m = re.search(r"POI\s+id\s+(\d+)", answer, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b", answer)
    return int(m.group(1)) if m else None


def bucket_from_time(tm: TimeBucketManager, hour: int, dow: int) -> int:
    return int(tm.get_bucket(int(hour) % 24, int(dow) >= 5))


def load_feature_cache(cache_dir: str, all_pois: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    sem_path = os.path.join(cache_dir, "poi_sem_vectors.npy")
    geo_path = os.path.join(cache_dir, "poi_geo_features.npy")
    poi_path = os.path.join(cache_dir, "poi_id_list.json")
    if not os.path.exists(sem_path) or not os.path.exists(geo_path) or not os.path.exists(poi_path):
        raise FileNotFoundError(
            f"Feature cache must contain poi_sem_vectors.npy, poi_geo_features.npy, poi_id_list.json: {cache_dir}"
        )
    with open(poi_path, "r", encoding="utf-8") as f:
        cached_pois = [int(x) for x in json.load(f)]
    if cached_pois != [int(x) for x in all_pois]:
        raise RuntimeError("Feature cache POI order does not match this dataset. Rebuild cache for the same dataset.")
    sem = torch.tensor(np.load(sem_path).astype(np.float32), dtype=torch.float32)
    geo = torch.tensor(np.load(geo_path).astype(np.float32), dtype=torch.float32)
    return sem, geo


class TemporalTransitionIndex:
    def __init__(self):
        self.edges = defaultdict(list)
        self.out_times = defaultdict(list)
        self.in_times = defaultdict(list)

    def add(self, epoch: int, bucket: int, src: int, dst: int):
        self.edges[(bucket, src, dst)].append(int(epoch))
        self.out_times[(bucket, src)].append((int(epoch), int(dst)))
        self.in_times[(bucket, dst)].append(int(epoch))

    def build(self):
        for key in self.edges:
            self.edges[key].sort()
        for key in self.out_times:
            self.out_times[key].sort(key=lambda x: x[0])
        for key in self.in_times:
            self.in_times[key].sort()

    @staticmethod
    def _count_before(sorted_epochs: List[int], cutoff: Optional[int]) -> int:
        if cutoff is None:
            return len(sorted_epochs)
        import bisect

        return bisect.bisect_left(sorted_epochs, int(cutoff))

    def edge_count(self, bucket: int, src: int, dst: int, cutoff: Optional[int]) -> int:
        return self._count_before(self.edges.get((bucket, src, dst), []), cutoff)

    def neighbors(self, bucket: int, src: int, cutoff: Optional[int]) -> Dict[int, int]:
        entries = self.out_times.get((bucket, src), [])
        out = defaultdict(int)
        for epoch, dst in entries:
            if cutoff is not None and epoch >= cutoff:
                break
            out[int(dst)] += 1
        return dict(out)

    def node_stats(self, bucket: int, node: int, cutoff: Optional[int]) -> Tuple[int, int, int, int]:
        out_nb = self.neighbors(bucket, node, cutoff)
        out_cnt = sum(out_nb.values())
        in_cnt = self._count_before(self.in_times.get((bucket, node), []), cutoff)
        return out_cnt, in_cnt, len(out_nb), max(out_nb.values()) if out_nb else 0


class SubgraphBuilder:
    def __init__(self, index: TemporalTransitionIndex, max_nodes: int = 20):
        self.index = index
        self.max_nodes = max_nodes

    def build(self, center: int, bucket: int, cutoff: Optional[int]) -> Data:
        nb = self.index.neighbors(bucket, center, cutoff)
        nodes = [int(center)]
        for dst, _ in sorted(nb.items(), key=lambda x: (-x[1], x[0]))[: self.max_nodes - 1]:
            if dst != center:
                nodes.append(int(dst))

        nmap = {p: i for i, p in enumerate(nodes)}
        srcs, dsts, weights = [], [], []
        for src in nodes:
            for dst in nodes:
                w = self.index.edge_count(bucket, src, dst, cutoff)
                if w > 0:
                    srcs.append(nmap[src])
                    dsts.append(nmap[dst])
                    weights.append(float(math.log1p(w)))
        if not srcs:
            srcs, dsts, weights = [0], [0], [1.0]

        stats = np.zeros((len(nodes), 4), dtype=np.float32)
        for i, node in enumerate(nodes):
            stats[i] = np.asarray(self.index.node_stats(bucket, node, cutoff), dtype=np.float32)
        stats = stats / (stats.max(axis=0, keepdims=True) + 1e-6)

        return Data(
            poi_idx=torch.tensor(nodes, dtype=torch.long),
            graph_stats=torch.tensor(stats, dtype=torch.float32),
            edge_index=torch.tensor([srcs, dsts], dtype=torch.long),
            edge_weight=torch.tensor(weights, dtype=torch.float32),
            time_bucket=torch.tensor([int(bucket)], dtype=torch.long),
            num_nodes=len(nodes),
        )


@dataclass
class RetrievalSample:
    query_poi_indices: List[int]
    query_hours: List[int]
    query_dows: List[int]
    target_poi_idx: int
    current_hour: int
    current_dow: int
    bucket: int
    target_epoch: int
    graph: Data


def build_transition_index(dataset, p2i: Dict[int, int], tm: TimeBucketManager) -> TemporalTransitionIndex:
    idx = TemporalTransitionIndex()
    for traj in dataset.all_trajectories.values():
        visits = getattr(traj, "visits", [])
        for i in range(len(visits) - 1):
            src_v = visits[i]
            dst_v = visits[i + 1]
            src = p2i.get(int(src_v.poi_id), -1)
            dst = p2i.get(int(dst_v.poi_id), -1)
            if src < 0 or dst < 0:
                continue
            idx.add(int(dst_v.epoch), int(tm.get_bucket_from_visit(dst_v)), src, dst)
    idx.build()
    return idx


def build_samples(
    qa_path: str,
    p2i: Dict[int, int],
    tm: TimeBucketManager,
    graph_builder: SubgraphBuilder,
    max_seq_len: int,
    max_samples: int,
) -> List[RetrievalSample]:
    rows = load_json_or_jsonl(qa_path)
    samples = []
    for row in tqdm(rows, desc=f"parse {os.path.basename(qa_path)}"):
        traj, target_epoch, target_hour, target_dow = parse_qa_sample(row.get("question", ""))
        target_poi = parse_answer_poi(row.get("answer", ""))
        if not traj or target_poi is None:
            continue
        if target_epoch is None or target_hour is None or target_dow is None:
            last = traj[-1]
            target_epoch = int(last["epoch"])
            target_hour = int(last["hour"])
            target_dow = int(last["dow"])

        target_idx = p2i.get(int(target_poi), -1)
        if target_idx < 0:
            continue

        visits = traj[-max_seq_len:] if max_seq_len > 0 else traj
        poi_indices = [p2i.get(int(v["poi_id"]), -1) for v in visits]
        if any(x < 0 for x in poi_indices):
            continue
        hours = [int(v["hour"]) % 24 for v in visits]
        dows = [int(v["dow"]) % 7 for v in visits]
        bucket = bucket_from_time(tm, target_hour, target_dow)
        graph = graph_builder.build(center=poi_indices[-1], bucket=bucket, cutoff=target_epoch)
        samples.append(
            RetrievalSample(
                query_poi_indices=poi_indices,
                query_hours=hours,
                query_dows=dows,
                target_poi_idx=target_idx,
                current_hour=int(target_hour) % 24,
                current_dow=int(target_dow) % 7,
                bucket=bucket,
                target_epoch=int(target_epoch),
                graph=graph,
            )
        )
        if max_samples and len(samples) >= max_samples:
            break
    print(f"Built {len(samples)} samples from {qa_path}")
    return samples


class MultiViewDataset(Dataset):
    def __init__(self, samples: List[RetrievalSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[RetrievalSample]) -> dict:
    max_len = max(len(x.query_poi_indices) for x in batch)

    def pad(values, pad_value=0):
        return F.pad(torch.tensor(values, dtype=torch.long), (0, max_len - len(values)), value=pad_value)

    return {
        "poi_seq": torch.stack([pad(x.query_poi_indices) for x in batch]),
        "hour_seq": torch.stack([pad(x.query_hours) for x in batch]),
        "dow_seq": torch.stack([pad(x.query_dows) for x in batch]),
        "lengths": torch.tensor([len(x.query_poi_indices) for x in batch], dtype=torch.long),
        "last_poi": torch.tensor([x.query_poi_indices[-1] for x in batch], dtype=torch.long),
        "current_hour": torch.tensor([x.current_hour for x in batch], dtype=torch.long),
        "current_dow": torch.tensor([x.current_dow for x in batch], dtype=torch.long),
        "bucket": torch.tensor([x.bucket for x in batch], dtype=torch.long),
        "target": torch.tensor([x.target_poi_idx for x in batch], dtype=torch.long),
        "graph": Batch.from_data_list([x.graph for x in batch]),
    }


class POIEncoder(nn.Module):
    def __init__(self, sem_vectors: torch.Tensor, geo_features: torch.Tensor, hidden_size: int, id_dim: int = 128):
        super().__init__()
        num_pois, sem_dim = sem_vectors.shape
        geo_dim = geo_features.shape[1]
        self.register_buffer("sem_vectors", sem_vectors)
        self.register_buffer("geo_features", geo_features)
        self.id_embedding = nn.Embedding(num_pois, id_dim)
        self.sem_proj = nn.Sequential(nn.Linear(sem_dim, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.geo_proj = nn.Sequential(nn.Linear(geo_dim, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.id_proj = nn.Sequential(nn.Linear(id_dim, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, poi_idx: torch.Tensor) -> torch.Tensor:
        sem = self.sem_proj(self.sem_vectors[poi_idx])
        geo = self.geo_proj(self.geo_features[poi_idx])
        pid = self.id_proj(self.id_embedding(poi_idx))
        return F.normalize(self.fuse(torch.cat([pid, sem, geo], dim=-1)), dim=-1)

    def all_embeddings(self) -> torch.Tensor:
        idx = torch.arange(self.sem_vectors.size(0), device=self.sem_vectors.device)
        return self.forward(idx)


class SemanticViewEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_buckets: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hour_emb = nn.Embedding(24, hidden_size)
        self.dow_emb = nn.Embedding(7, hidden_size)
        self.bucket_emb = nn.Embedding(num_buckets, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, last_poi_emb, hours, dows, buckets):
        x = torch.cat([last_poi_emb, self.hour_emb(hours), self.dow_emb(dows), self.bucket_emb(buckets)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TrajectoryTransformerEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hour_emb = nn.Embedding(24, hidden_size)
        self.dow_emb = nn.Embedding(7, hidden_size)
        self.in_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))

    def forward(self, poi_emb_seq, hours, dows, lengths):
        x = self.in_proj(torch.cat([poi_emb_seq, self.hour_emb(hours), self.dow_emb(dows)], dim=-1))
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.encoder(x, src_key_padding_mask=mask)
        last_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last = h.gather(1, last_idx).squeeze(1)
        return F.normalize(self.out(last), dim=-1)


class StructureGATEncoder(nn.Module):
    def __init__(self, hidden_size: int, graph_stat_dim: int = 4, num_buckets: int = 8, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.bucket_emb = nn.Embedding(num_buckets, hidden_size)
        self.stat_proj = nn.Sequential(nn.Linear(graph_stat_dim, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.input_proj = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.gat1 = GATConv(hidden_size, hidden_size // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_size, hidden_size, heads=1, dropout=dropout)
        self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))

    def forward(self, graph: Batch, node_poi_emb: torch.Tensor):
        bucket = graph.time_bucket[graph.batch]
        x = torch.cat([node_poi_emb, self.stat_proj(graph.graph_stats), self.bucket_emb(bucket)], dim=-1)
        h = self.input_proj(x)
        h = F.elu(self.gat1(h, graph.edge_index))
        h = self.gat2(h, graph.edge_index)
        return F.normalize(self.out(global_mean_pool(h, graph.batch)), dim=-1)


class GatedFusion(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, z_sem, z_str, z_traj):
        gate = F.softmax(self.gate(torch.cat([z_sem, z_str, z_traj], dim=-1)), dim=-1)
        z = gate[:, 0:1] * z_sem + gate[:, 1:2] * z_str + gate[:, 2:3] * z_traj
        return F.normalize(z, dim=-1), gate


class MultiViewPOIRetriever(nn.Module):
    def __init__(self, sem_vectors: torch.Tensor, geo_features: torch.Tensor, hidden_size: int, dropout: float):
        super().__init__()
        self.poi_encoder = POIEncoder(sem_vectors, geo_features, hidden_size)
        self.semantic_encoder = SemanticViewEncoder(hidden_size, dropout=dropout)
        self.trajectory_encoder = TrajectoryTransformerEncoder(hidden_size, dropout=dropout)
        self.structure_encoder = StructureGATEncoder(hidden_size, dropout=dropout)
        self.fusion = GatedFusion(hidden_size, dropout=dropout)

    def forward(self, batch: dict):
        poi_seq = batch["poi_seq"]
        z_seq = self.poi_encoder(poi_seq)
        z_last = self.poi_encoder(batch["last_poi"])
        graph = batch["graph"]
        node_poi_emb = self.poi_encoder(graph.poi_idx)

        z_sem = self.semantic_encoder(z_last, batch["current_hour"], batch["current_dow"], batch["bucket"])
        z_traj = self.trajectory_encoder(z_seq, batch["hour_seq"], batch["dow_seq"], batch["lengths"])
        z_str = self.structure_encoder(graph, node_poi_emb)
        z_fused, gate = self.fusion(z_sem, z_str, z_traj)
        return {"sem": z_sem, "str": z_str, "traj": z_traj, "fused": z_fused, "gate": gate}


def retrieval_ce_loss(z: torch.Tensor, all_poi: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z @ all_poi.t() / temperature
    return F.cross_entropy(logits, target)


def hard_negative_margin_loss(
    z: torch.Tensor,
    all_poi: torch.Tensor,
    target: torch.Tensor,
    margin: float,
    hard_topk: int,
) -> torch.Tensor:
    if hard_topk <= 0:
        return z.new_tensor(0.0)
    scores = z @ all_poi.t()
    pos_scores = scores.gather(1, target.view(-1, 1))
    neg_scores = scores.masked_fill(
        F.one_hot(target, num_classes=scores.size(1)).bool(),
        float("-inf"),
    )
    k = min(int(hard_topk), max(1, scores.size(1) - 1))
    hard_scores = neg_scores.topk(k, dim=-1).values
    return F.relu(float(margin) + hard_scores - pos_scores).mean()


def contrastive_loss(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def compute_loss(model: MultiViewPOIRetriever, out: dict, target: torch.Tensor, args) -> Tuple[torch.Tensor, dict]:
    all_poi = model.poi_encoder.all_embeddings()
    loss_fused = retrieval_ce_loss(out["fused"], all_poi, target, args.temperature)
    loss_traj = retrieval_ce_loss(out["traj"], all_poi, target, args.temperature)
    loss_sem = retrieval_ce_loss(out["sem"], all_poi, target, args.temperature)
    loss_str = retrieval_ce_loss(out["str"], all_poi, target, args.temperature)
    loss_align = (
        contrastive_loss(out["sem"], out["traj"], args.align_temperature)
        + contrastive_loss(out["str"], out["traj"], args.align_temperature)
        + 0.5 * contrastive_loss(out["sem"], out["str"], args.align_temperature)
    )
    loss_rank = hard_negative_margin_loss(
        out["fused"],
        all_poi,
        target,
        args.rank_margin,
        args.hard_topk,
    )
    loss = (
        loss_fused
        + args.traj_weight * loss_traj
        + args.sem_weight * loss_sem
        + args.str_weight * loss_str
        + args.align_weight * loss_align
        + args.rank_weight * loss_rank
    )
    return loss, {
        "loss": float(loss.detach().item()),
        "loss_fused": float(loss_fused.detach().item()),
        "loss_traj": float(loss_traj.detach().item()),
        "loss_sem": float(loss_sem.detach().item()),
        "loss_str": float(loss_str.detach().item()),
        "loss_align": float(loss_align.detach().item()),
        "loss_rank": float(loss_rank.detach().item()),
        "gate_sem": float(out["gate"][:, 0].detach().mean().item()),
        "gate_str": float(out["gate"][:, 1].detach().mean().item()),
        "gate_traj": float(out["gate"][:, 2].detach().mean().item()),
    }


@torch.no_grad()
def evaluate(model: MultiViewPOIRetriever, loader: DataLoader, device: torch.device, top_ks=(1, 5, 10, 20)) -> dict:
    model.eval()
    hit = {k: 0 for k in top_ks}
    total = 0
    all_poi = model.poi_encoder.all_embeddings()
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        scores = out["fused"] @ all_poi.t()
        top = scores.topk(max(top_ks), dim=-1).indices
        target = batch["target"]
        for i in range(target.size(0)):
            total += 1
            pred = top[i].tolist()
            tgt = int(target[i].item())
            for k in top_ks:
                if tgt in pred[:k]:
                    hit[k] += 1
    metrics = {f"recall@{k}": hit[k] / max(total, 1) for k in top_ks}
    metrics["total"] = total
    return metrics


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    tm = TimeBucketManager()

    dataset = load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa)
    all_pois = sorted(int(x) for x in dataset.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
    sem_vectors, geo_features = load_feature_cache(args.feature_cache, all_pois)

    transition_index = build_transition_index(dataset, p2i, tm)
    graph_builder = SubgraphBuilder(transition_index, max_nodes=args.max_graph_nodes)
    train_samples = build_samples(args.train_qa, p2i, tm, graph_builder, args.max_seq_len, args.max_train_samples)
    val_samples = build_samples(args.test_qa, p2i, tm, graph_builder, args.max_seq_len, args.max_val_samples)
    if not train_samples:
        raise RuntimeError("No train samples were built.")
    if not val_samples:
        raise RuntimeError("No validation samples were built.")

    train_loader = DataLoader(
        MultiViewDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        MultiViewDataset(val_samples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = MultiViewPOIRetriever(sem_vectors, geo_features, args.hidden_size, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best_recall20 = -1.0
    history = []
    print(f"[target-poi-multiview] train={len(train_samples)} val={len(val_samples)} pois={len(all_pois)}")
    print(f"[target-poi-multiview] hidden={args.hidden_size} align_weight={args.align_weight}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = defaultdict(float)
        steps = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            batch = move_batch(batch, device)
            opt.zero_grad()
            out = model(batch)
            loss, logs = compute_loss(model, out, batch["target"], args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            for k, v in logs.items():
                sums[k] += float(v)
            steps += 1
        sched.step()

        train_logs = {k: v / max(1, steps) for k, v in sums.items()}
        val_metrics = evaluate(model, val_loader, device=device)
        row = {"epoch": epoch, **train_logs, **val_metrics}
        history.append(row)
        print(
            f"[target-poi-multiview] epoch={epoch} "
            f"loss={row['loss']:.4f} fused={row['loss_fused']:.4f} "
            f"rank={row['loss_rank']:.4f} align={row['loss_align']:.4f} "
            f"recall@1={row['recall@1']:.4f} recall@5={row['recall@5']:.4f} "
            f"recall@10={row['recall@10']:.4f} recall@20={row['recall@20']:.4f} "
            f"gate=({row['gate_sem']:.2f},{row['gate_str']:.2f},{row['gate_traj']:.2f})"
        )

        if row["recall@20"] > best_recall20:
            best_recall20 = row["recall@20"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pth"))

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, "poi_id_list.json"), "w", encoding="utf-8") as f:
        json.dump(all_pois, f, ensure_ascii=False)
    torch.save(vars(args), os.path.join(args.save_dir, "config.pth"))
    print(f"[target-poi-multiview] best recall@20={best_recall20:.4f}; saved to {args.save_dir}")


def build_parser():
    default_dataset = os.getenv("DATASET_NAME", "nyc")
    default_data_dir = f"./datasets/{default_dataset}/preprocessed"
    p = argparse.ArgumentParser(description="Direct target-POI retrieval with semantic/structure/trajectory views.")
    p.add_argument("--dataset_name", default=default_dataset)
    p.add_argument("--train_csv", default=f"{default_data_dir}/train_sample.csv")
    p.add_argument("--test_csv", default=f"{default_data_dir}/test_sample_with_traj.csv")
    p.add_argument("--train_qa", default=f"{default_data_dir}/train_qa_pairs_kqt.json")
    p.add_argument("--test_qa", default=f"{default_data_dir}/test_qa_pairs_kqt.json")
    p.add_argument("--feature_cache", default="./rag/feature_cache/bert")
    p.add_argument("--save_dir", default="./rag/target_poi_multiview/artifacts")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=20)
    p.add_argument("--max_graph_nodes", type=int, default=20)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--align_temperature", type=float, default=0.07)
    p.add_argument("--traj_weight", type=float, default=0.1)
    p.add_argument("--sem_weight", type=float, default=0.05)
    p.add_argument("--str_weight", type=float, default=0.05)
    p.add_argument("--align_weight", type=float, default=0.0)
    p.add_argument("--rank_weight", type=float, default=0.0)
    p.add_argument("--rank_margin", type=float, default=0.1)
    p.add_argument("--hard_topk", type=int, default=20)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
