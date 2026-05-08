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


class TemporalEdgeIndex:
    """Index by (bucket, source_idx) -> sorted [(epoch, dst_idx), ...]."""

    def __init__(self):
        self.by_bucket_source = defaultdict(list)
        self._epoch_cache = {}

    def add(self, epoch, src_idx, dst_idx, bucket):
        self.by_bucket_source[(bucket, src_idx)].append((epoch, dst_idx))

    def build(self):
        for k in self.by_bucket_source:
            self.by_bucket_source[k].sort(key=lambda x: x[0])
            self._epoch_cache[k] = [e for e, _ in self.by_bucket_source[k]]

    def next_hops_before(self, src_idx, bucket, cutoff_epoch):
        entries = self.by_bucket_source.get((bucket, src_idx), [])
        if not entries:
            return {}
        epochs = self._epoch_cache[(bucket, src_idx)]
        n = bisect.bisect_left(epochs, cutoff_epoch)
        counts = defaultdict(int)
        for i in range(n):
            counts[entries[i][1]] += 1
        return dict(counts)

    def next_hops_all(self, src_idx, bucket):
        entries = self.by_bucket_source.get((bucket, src_idx), [])
        counts = defaultdict(int)
        for _, dst_idx in entries:
            counts[dst_idx] += 1
        return dict(counts)


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


def get_poi_name(poi_id, poi_dict):
    poi = poi_dict.get(poi_id)
    if poi is None:
        return f"POI_{poi_id}"
    return getattr(poi, "name", f"POI_{poi_id}")


def get_poi_category(poi_id, poi_dict):
    poi = poi_dict.get(poi_id)
    if poi is None:
        return "Unknown"
    return getattr(poi, "category_name", "Unknown")


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_semantic_prompt(retrieved_data, all_pois, poi_dict, per_center_hops=8):
    if not retrieved_data:
        return ""

    bucket_names = {
        0: "Weekday Morning (6-11)",
        1: "Weekday Noon (11-14)",
        2: "Weekday Afternoon (14-18)",
        3: "Weekday Evening (18-6)",
        4: "Weekend Morning (6-11)",
        5: "Weekend Noon (11-14)",
        6: "Weekend Afternoon (14-18)",
        7: "Weekend Evening (18-6)",
    }

    lines = [
        "[Semantic Transition Evidence]",
        "These transitions are retrieved using the last POI and current time context.",
        "They represent similar next-hop patterns and indicate likely next POIs.",
        "You can directly use these transitions as reference for prediction.",
        "",
        "For each center POI under the given time context, its next-hop transitions are",
        "(distances are measured from the center POI to each next-hop POI):",
        "",
    ]

    for item in retrieved_data:
        center_poi_id = item["center_poi"]
        center_name = get_poi_name(center_poi_id, poi_dict)
        bucket = item["bucket"]
        bucket_label = bucket_names.get(bucket, f"Time period {bucket}")
        next_hops = item["next_hops"]
        if not next_hops:
            continue

        lines.append(f"At {bucket_label}, for POI {center_poi_id} ({center_name}):")

        hop_items = []
        sorted_hops = sorted(next_hops.items(), key=lambda x: -x[1])
        for next_idx, _ in sorted_hops[:per_center_hops]:
            if next_idx >= len(all_pois):
                continue
            next_poi_id = all_pois[next_idx]
            next_cat = get_poi_category(next_poi_id, poi_dict)

            dist_str = "unknown"
            c_poi = poi_dict.get(center_poi_id)
            n_poi = poi_dict.get(next_poi_id)
            if c_poi and n_poi:
                try:
                    dist_km = _haversine(c_poi.latitude, c_poi.longitude, n_poi.latitude, n_poi.longitude)
                    dist_str = f"{dist_km:.1f}"
                except Exception:
                    pass

            hop_items.append(f"POI {next_poi_id} (category: {next_cat}, distance: {dist_str} km)")

        if hop_items:
            lines.append("Next-hop POIs: " + ", ".join(hop_items))
            lines.append("")

    if len(lines) <= 8:
        return ""
    return "\n".join(lines)


def _get_train_trajectories(dataset):
    for attr in ["train_trajectories", "train_trajs", "trajectories_train"]:
        val = getattr(dataset, attr, None)
        if isinstance(val, dict):
            print(f"Using dataset.{attr} as retrieval fact source")
            return val, attr
    print("dataset.train_trajectories not found; fallback to dataset.all_trajectories")
    return dataset.all_trajectories, "all_trajectories"


def build_edge_index(dataset, p2i, tm):
    traj_dict, source_attr = _get_train_trajectories(dataset)
    ei = TemporalEdgeIndex()
    n_edges = 0

    for traj in traj_dict.values():
        visits = getattr(traj, "visits", [])
        for i in range(len(visits) - 1):
            s = visits[i]
            d = visits[i + 1]
            si = p2i.get(s.poi_id, -1)
            di = p2i.get(d.poi_id, -1)
            if si < 0 or di < 0:
                continue
            bucket = tm.get_bucket_from_visit(s)
            ei.add(epoch=s.epoch, src_idx=si, dst_idx=di, bucket=bucket)
            n_edges += 1

    ei.build()
    print(f"Temporal edge index built from {source_attr}: edges={n_edges}, source_keys={len(ei.by_bucket_source)}")
    return ei


def retrieve_for_sample(
    trajectory,
    target_epoch,
    all_pois,
    p2i,
    sem_enc,
    sem_all,
    cand_emb,
    edge_index,
    tm,
    top_k=10,
    use_masked=True,
    device="cpu",
):
    if not trajectory:
        return []

    dev = torch.device(device)
    last = trajectory[-1]
    query_poi_id = last["poi_id"]
    query_idx = p2i.get(query_poi_id, -1)
    if query_idx < 0:
        return []

    hour = last["hour"]
    is_weekend = last["day_of_week"] >= 5
    bucket = tm.get_bucket(hour, is_weekend)

    with torch.no_grad():
        q_llm = sem_all[query_idx : query_idx + 1]
        b_t = torch.tensor([bucket], dtype=torch.long, device=dev)
        z_q = sem_enc.encode_query(q_llm, b_t)
        scores = (z_q @ cand_emb.t()).squeeze(0)
        k = min(top_k, scores.numel())
        top_vals, top_ids = torch.topk(scores, k=k)

    retrieved = []
    for rank, (score, src_idx) in enumerate(zip(top_vals.tolist(), top_ids.tolist()), start=1):
        if use_masked and target_epoch is not None:
            hops = edge_index.next_hops_before(src_idx=src_idx, bucket=bucket, cutoff_epoch=target_epoch)
        else:
            hops = edge_index.next_hops_all(src_idx=src_idx, bucket=bucket)

        retrieved.append(
            {
                "rank": rank,
                "score": float(score),
                "bucket": bucket,
                "center_poi": all_pois[src_idx],
                "center_idx": src_idx,
                "next_hops": hops,
            }
        )
    return retrieved


def evaluate_hit(retrieved, answer_poi_id, all_pois, top_ks=(10, 20, 50)):
    if answer_poi_id is None:
        return {f"hit@{k}": False for k in top_ks}

    out = {}
    for k in top_ks:
        center_ids = {x["center_poi"] for x in retrieved[:k]}
        hop_ids = set()
        for x in retrieved[:k]:
            for dst_idx in x["next_hops"].keys():
                if 0 <= dst_idx < len(all_pois):
                    hop_ids.add(all_pois[dst_idx])
        out[f"hit@{k}"] = answer_poi_id in center_ids or answer_poi_id in hop_ids
    return out


def process_qa_file(
    input_path,
    output_path,
    split_name,
    all_pois,
    p2i,
    poi_dict,
    sem_enc,
    sem_all,
    cand_emb,
    edge_index,
    tm,
    top_k=10,
    per_center_hops=8,
    use_masked=True,
    device="cpu",
):
    qa_data = load_qa_data(input_path)
    print(f"Processing {split_name}: {len(qa_data)} samples")

    top_ks = (10, 20, 50)
    hit_counts = {f"hit@{k}": 0 for k in top_ks}
    valid_eval = 0
    with_prompt = 0

    enriched = []
    for row in tqdm(qa_data, desc=f"Retrieving {split_name}"):
        row = row.copy()
        q = row.get("question", "")
        a = row.get("answer", "")

        traj, target_epoch = parse_qa_sample(q)
        answer_poi = parse_answer_poi(a)

        retrieved = retrieve_for_sample(
            trajectory=traj,
            target_epoch=target_epoch,
            all_pois=all_pois,
            p2i=p2i,
            sem_enc=sem_enc,
            sem_all=sem_all,
            cand_emb=cand_emb,
            edge_index=edge_index,
            tm=tm,
            top_k=top_k,
            use_masked=use_masked,
            device=device,
        )

        prompt = build_semantic_prompt(retrieved, all_pois, poi_dict, per_center_hops=per_center_hops)
        row["question_base"] = q
        row["retrieval_text_sem"] = prompt
        row["retrieval_items_sem"] = retrieved
        if prompt:
            row["question"] = q + prompt
            with_prompt += 1

        if answer_poi is not None:
            hits = evaluate_hit(retrieved, answer_poi, all_pois, top_ks=top_ks)
            for k in top_ks:
                if hits[f"hit@{k}"]:
                    hit_counts[f"hit@{k}"] += 1
            valid_eval += 1

        enriched.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"Saved {split_name} enriched QA to {output_path}")
    if valid_eval > 0:
        for k in top_ks:
            rate = hit_counts[f"hit@{k}"] / valid_eval
            print(f"  {split_name} hit@{k}: {hit_counts[f'hit@{k}']} / {valid_eval} = {rate:.4f}")
    print(f"  {split_name} prompts added: {with_prompt} / {len(qa_data)}")

    return {
        "split": split_name,
        "total": len(qa_data),
        "valid_eval": valid_eval,
        "with_prompt": with_prompt,
        **{k: hit_counts[k] for k in hit_counts},
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="./datasets/nyc/preprocessed/train_sample.csv")
    p.add_argument("--test_csv", default="./datasets/nyc/preprocessed/test_sample.csv")
    p.add_argument("--train_qa", default="./datasets/nyc/preprocessed/train_qa_pairs_kqt.json")
    p.add_argument("--test_qa", default="./datasets/nyc/preprocessed/test_qa_pairs_kqt.json")
    p.add_argument("--sem_dir", default="./sem_lib/")
    p.add_argument("--output_dir", default="./rag/sem_enriched_qa/")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--per_center_hops", type=int, default=2)
    p.add_argument("--encoder", default="bert")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    ds = load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa)
    tm = TimeBucketManager()

    all_pois = sorted(ds.poi_dict.keys())
    p2i = {p: i for i, p in enumerate(all_pois)}
    n_pois = len(all_pois)

    print("Loading semantic features...")
    try:
        from train_semantic_view import prepare_features

        fd = prepare_features(ds, None, all_pois, args.encoder, args.device, True)
    except Exception:
        sem_v = np.random.randn(n_pois, 32).astype(np.float32)
        fd = {"sem_vectors": sem_v, "llm_dim": 32}

    sem_all = torch.tensor(fd["sem_vectors"], dtype=torch.float32, device=dev)

    print("Loading semantic encoder...")
    config = torch.load(os.path.join(args.sem_dir, "config.pth"), map_location=dev)
    shared_dim = int(config["shared_dim"])
    num_buckets = int(config["num_buckets"])

    sem_enc = SemanticEncoder(
        llm_dim=int(fd["llm_dim"]),
        num_buckets=num_buckets,
        hidden=512,
        shared_dim=shared_dim,
    ).to(dev)
    sem_enc.load_state_dict(torch.load(os.path.join(args.sem_dir, "sem_encoder.pth"), map_location=dev))
    sem_enc.eval()

    with torch.no_grad():
        cand_emb = sem_enc.encode_candidate(sem_all)

    print("Building temporal edge index from train trajectories...")
    edge_index = build_edge_index(ds, p2i, tm)

    summary = {
        "top_k": args.top_k,
        "per_center_hops": args.per_center_hops,
        "fact_source": config.get("fact_source", "unknown"),
        "splits": [],
    }

    if os.path.exists(args.train_qa):
        train_out = os.path.join(args.output_dir, "train_qa_sem_retrieved.json")
        stats = process_qa_file(
            input_path=args.train_qa,
            output_path=train_out,
            split_name="train",
            all_pois=all_pois,
            p2i=p2i,
            poi_dict=ds.poi_dict,
            sem_enc=sem_enc,
            sem_all=sem_all,
            cand_emb=cand_emb,
            edge_index=edge_index,
            tm=tm,
            top_k=args.top_k,
            per_center_hops=args.per_center_hops,
            use_masked=True,
            device=args.device,
        )
        summary["splits"].append(stats)

    if os.path.exists(args.test_qa):
        test_out = os.path.join(args.output_dir, "test_qa_sem_retrieved.json")
        stats = process_qa_file(
            input_path=args.test_qa,
            output_path=test_out,
            split_name="test",
            all_pois=all_pois,
            p2i=p2i,
            poi_dict=ds.poi_dict,
            sem_enc=sem_enc,
            sem_all=sem_all,
            cand_emb=cand_emb,
            edge_index=edge_index,
            tm=tm,
            top_k=args.top_k,
            per_center_hops=args.per_center_hops,
            use_masked=False,
            device=args.device,
        )
        summary["splits"].append(stats)

    summary_path = os.path.join(args.output_dir, "retrieval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
