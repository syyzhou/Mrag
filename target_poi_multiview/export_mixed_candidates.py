import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import train as mv  # noqa: E402


VISIT_RE = re.compile(
    r"At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+"
    r"user\s+\d+\s+visited\s+POI\s+id\s+(\d+)\s+"
    r"which\s+is\s+a\s+(.+?)\s+with\s+Category\s+id\s+(\d+)"
)
ANSWER_RE = re.compile(r"will visit POI id\s+(\d+)\.([^\.\n]+)", re.IGNORECASE)


def load_rows(path):
    if path.lower().endswith(".json"):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "<answer>:" in line:
            q, a = line.split("<answer>:", 1)
            rows.append({"question": q.replace("<question>:", "").strip(), "answer": a.strip()})
        else:
            rows.append(json.loads(line))
    return rows


def strip_existing_candidate_text(question):
    marker = "Use the following candidate POIs as supplementary references"
    idx = question.find(marker)
    return question[:idx].rstrip() if idx >= 0 else question


def parse_target(answer):
    return mv.parse_answer_poi(answer)


def parse_last_poi(question):
    traj, _target_epoch, _target_hour, _target_dow = mv.parse_qa_sample(question)
    return int(traj[-1]["poi_id"]) if traj else None


def load_poi_info(poi_info_csv, poi_desc_csv, train_rows, test_rows):
    poi_info = defaultdict(lambda: {"lat": None, "lng": None, "category": "Unknown", "cat_id": None})

    if poi_info_csv and os.path.exists(poi_info_csv):
        df = pd.read_csv(poi_info_csv)
        for row in df.itertuples(index=False):
            data = row._asdict()
            pid = int(data.get("poi_id"))
            poi_info[pid]["lat"] = float(data.get("latitude"))
            poi_info[pid]["lng"] = float(data.get("longitude"))
            if "poi_catid" in data:
                poi_info[pid]["cat_id"] = int(data.get("poi_catid"))

    if poi_desc_csv and os.path.exists(poi_desc_csv):
        df = pd.read_csv(poi_desc_csv)
        for row in df.itertuples(index=False):
            data = row._asdict()
            pid = int(data.get("poi_id"))
            poi_info[pid]["category"] = str(data.get("PoiCategoryName", "Unknown")).strip() or "Unknown"

    for rows in (train_rows, test_rows):
        for item in rows:
            text = item.get("question", "") + " " + item.get("answer", "")
            for _ts, poi_id, category, cat_id in VISIT_RE.findall(text):
                pid = int(poi_id)
                poi_info[pid]["category"] = category.strip()
                poi_info[pid]["cat_id"] = int(cat_id)
            m = ANSWER_RE.search(item.get("answer", ""))
            if m:
                poi_info[int(m.group(1))]["category"] = m.group(2).strip()
    return poi_info


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(a))


def format_distance(distance_km):
    if distance_km < 1.0:
        return f"{distance_km * 1000:.0f}m"
    if distance_km < 10.0:
        return f"{distance_km:.1f}km"
    return f"{distance_km:.0f}km"


def build_candidate_text(candidate_ids, last_poi, poi_info):
    last_info = poi_info.get(last_poi) if last_poi is not None else None
    can_measure = bool(last_info and last_info.get("lat") is not None and last_info.get("lng") is not None)
    parts = []
    for poi_id in candidate_ids:
        info = poi_info.get(int(poi_id), {})
        category = info.get("category") or "Unknown"
        if can_measure and info.get("lat") is not None and info.get("lng") is not None:
            dist = haversine_km(last_info["lat"], last_info["lng"], info["lat"], info["lng"])
            parts.append(f"POI {int(poi_id)}({category}, {format_distance(dist)})")
        else:
            parts.append(f"POI {int(poi_id)}({category})")

    if can_measure:
        prefix = (
            "Use the following candidate POIs as supplementary references to refine your prediction. "
            "Each candidate is shown as POI id(category, distance), where category is the POI's "
            "functional attribute and distance is measured from the last visited "
            f"POI {last_poi} in the current trajectory. "
        )
    else:
        prefix = (
            "Use the following candidate POIs as supplementary references to refine your prediction. "
            "Each candidate is shown as POI id(category), where category is the POI's functional attribute. "
        )
    return prefix + "Candidate POIs:[" + ", ".join(parts) + "]."


def append_candidates(question, candidate_ids, last_poi, poi_info):
    base = strip_existing_candidate_text(question).rstrip()
    sep = " " if base.endswith(".") else ". "
    return base + sep + build_candidate_text(candidate_ids, last_poi, poi_info)


def build_export_samples(raw_rows, p2i, tm, graph_builder, max_seq_len):
    samples, raw_indices, skipped = [], [], []
    for idx, row in enumerate(tqdm(raw_rows, desc="build export samples")):
        question = row.get("question", "")
        traj, target_epoch, target_hour, target_dow = mv.parse_qa_sample(question)
        if not traj:
            skipped.append(idx)
            continue
        if target_epoch is None or target_hour is None or target_dow is None:
            last = traj[-1]
            target_epoch = int(last["epoch"])
            target_hour = int(last["hour"])
            target_dow = int(last["dow"])

        visits = traj[-max_seq_len:] if max_seq_len > 0 else traj
        poi_indices = [p2i.get(int(v["poi_id"]), -1) for v in visits]
        if any(x < 0 for x in poi_indices):
            skipped.append(idx)
            continue

        target_poi = parse_target(row.get("answer", ""))
        target_idx = p2i.get(int(target_poi), 0) if target_poi is not None else 0
        bucket = mv.bucket_from_time(tm, target_hour, target_dow)
        graph = graph_builder.build(center=poi_indices[-1], bucket=bucket, cutoff=int(target_epoch))
        samples.append(
            mv.RetrievalSample(
                query_poi_indices=poi_indices,
                query_hours=[int(v["hour"]) % 24 for v in visits],
                query_dows=[int(v["dow"]) % 7 for v in visits],
                target_poi_idx=target_idx,
                current_hour=int(target_hour) % 24,
                current_dow=int(target_dow) % 7,
                bucket=bucket,
                target_epoch=int(target_epoch),
                graph=graph,
            )
        )
        raw_indices.append(idx)
    return samples, raw_indices, skipped


@torch.no_grad()
def retrieve_candidates(model, loader, idx_to_poi, device, top_k):
    model.eval()
    all_poi = model.poi_encoder.all_embeddings()
    rows = []
    for batch in tqdm(loader, desc="retrieve"):
        batch = mv.move_batch(batch, device)
        scores = model(batch)["fused"] @ all_poi.t()
        vals, inds = scores.topk(top_k, dim=-1)
        for b in range(inds.size(0)):
            rows.append(
                [
                    {"poi_id": int(idx_to_poi[int(inds[b, j].item())]), "score": float(vals[b, j].item()), "rank": j + 1}
                    for j in range(inds.size(1))
                ]
            )
    return rows


def build_category_index(poi_info, all_pois):
    by_cat = defaultdict(list)
    for pid in all_pois:
        info = poi_info.get(int(pid), {})
        key = info.get("cat_id")
        if key is None:
            key = info.get("category") or "Unknown"
        by_cat[key].append(int(pid))
    return by_cat


def build_near_index(poi_info, all_pois):
    coords = []
    for pid in all_pois:
        info = poi_info.get(int(pid), {})
        if info.get("lat") is not None and info.get("lng") is not None:
            coords.append((int(pid), float(info["lat"]), float(info["lng"])))
    return coords


def add_unique(out, seen, poi_id, source, score=None, original_rank=None):
    poi_id = int(poi_id)
    if poi_id in seen:
        return False
    out.append({"poi_id": poi_id, "source": source, "score": score, "original_rank": original_rank})
    seen.add(poi_id)
    return True


def sample_same_category(target, poi_info, category_index, rng, limit, seen):
    info = poi_info.get(int(target), {})
    key = info.get("cat_id")
    if key is None:
        key = info.get("category") or "Unknown"
    pool = [p for p in category_index.get(key, []) if p != int(target) and p not in seen]
    rng.shuffle(pool)
    return pool[:limit]


def nearest_pois(last_poi, poi_info, coords, limit, seen, exclude):
    last = poi_info.get(int(last_poi), {}) if last_poi is not None else {}
    if last.get("lat") is None or last.get("lng") is None:
        return []
    ranked = []
    for pid, lat, lng in coords:
        if pid in seen or pid in exclude:
            continue
        ranked.append((haversine_km(last["lat"], last["lng"], lat, lng), pid))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [pid for _dist, pid in ranked[:limit]]


def build_train_mixed_candidates(
    row,
    retrieval_rows,
    all_pois,
    poi_info,
    category_index,
    coords,
    popular_pois,
    rng,
    args,
):
    target = parse_target(row.get("answer", ""))
    last_poi = parse_last_poi(row.get("question", ""))
    out, seen = [], set()

    retrieval_pool = list(retrieval_rows[: args.retrieval_pool_k])
    rng.shuffle(retrieval_pool)
    for cand in retrieval_pool:
        if len([x for x in out if x["source"] == "multiview"]) >= args.train_multiview_k:
            break
        add_unique(out, seen, cand["poi_id"], "multiview", cand.get("score"), cand.get("rank"))

    if target is not None:
        for pid in sample_same_category(target, poi_info, category_index, rng, args.same_category_k, seen):
            add_unique(out, seen, pid, "same_category")

    exclude = set(seen)
    if target is not None:
        exclude.add(int(target))
    for pid in nearest_pois(last_poi, poi_info, coords, args.near_k, seen, exclude):
        add_unique(out, seen, pid, "near")

    easy_pool = [p for p in popular_pois if p not in seen and (target is None or p != int(target))]
    if len(easy_pool) < args.easy_k:
        rest = [p for p in all_pois if p not in seen and (target is None or p != int(target))]
        rng.shuffle(rest)
        easy_pool.extend(rest)
    for pid in easy_pool[: args.easy_k]:
        add_unique(out, seen, pid, "popular_or_random")

    if len(out) < args.top_k:
        rest = [p for p in all_pois if p not in seen]
        rng.shuffle(rest)
        for pid in rest:
            add_unique(out, seen, pid, "random_fill")
            if len(out) >= args.top_k:
                break

    out = out[: args.top_k]
    rng.shuffle(out)
    for rank, cand in enumerate(out, start=1):
        cand["rank"] = rank
    return out


def build_test_topk_candidates(retrieval_rows, top_k):
    out = []
    seen = set()
    for cand in retrieval_rows:
        if add_unique(out, seen, cand["poi_id"], "multiview", cand.get("score"), cand.get("rank")) and len(out) >= top_k:
            break
    for rank, cand in enumerate(out, start=1):
        cand["rank"] = rank
    return out


def build_augmented_rows(raw_rows, candidate_map, poi_info):
    out = []
    stats = {
        "samples": len(raw_rows),
        "contains_target": 0,
        "contains_target_ratio": 0.0,
        "mrr_at_20": 0.0,
        "mrr_on_hit": 0.0,
        "bad_candidate_size": 0,
        "source_counts": Counter(),
    }
    rr_sum = 0.0
    rr_hit_sum = 0.0
    hit = 0
    for raw_idx, item in enumerate(raw_rows):
        cands = candidate_map.get(raw_idx, [])
        ids = [int(x["poi_id"]) for x in cands]
        target = parse_target(item.get("answer", ""))
        if target in ids:
            hit += 1
            rank = ids.index(target) + 1
            rr = 1.0 / rank
            rr_sum += rr
            rr_hit_sum += rr
        if len(ids) != 20:
            stats["bad_candidate_size"] += 1
        for cand in cands:
            stats["source_counts"][cand.get("source", "unknown")] += 1
        last_poi = parse_last_poi(item.get("question", ""))
        question = append_candidates(item["question"], ids, last_poi, poi_info) if ids else item["question"]
        out.append({"question": question, "answer": item["answer"], "retrieval_candidates_target_poi_multiview": cands})

    stats["contains_target"] = hit
    stats["contains_target_ratio"] = hit / len(raw_rows) if raw_rows else 0.0
    stats["mrr_at_20"] = rr_sum / len(raw_rows) if raw_rows else 0.0
    stats["mrr_on_hit"] = rr_hit_sum / hit if hit else 0.0
    stats["source_counts"] = dict(stats["source_counts"])
    return out, stats


def save_json(rows, path):
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def save_txt(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in rows:
            q = item["question"]
            a = item["answer"]
            if not q.startswith("<question>:"):
                q = "<question>: " + q
            if not a.startswith("<answer>:"):
                a = "<answer>: " + a
            f.write(q + " " + a + "\n")


def main():
    default_dataset = os.getenv("DATASET_NAME", "nyc")
    default_data_dir = f"./datasets/{default_dataset}/preprocessed"
    default_tag = "dst_time_mix8cat4near4easy4"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=default_dataset)
    parser.add_argument("--candidate_tag", default=default_tag)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--train_qa", default=None)
    parser.add_argument("--test_qa", default=None)
    parser.add_argument("--test_qa_for_dataset", default=None)
    parser.add_argument("--poi_info", default=None)
    parser.add_argument("--poi_desc", default=None)
    parser.add_argument("--feature_cache", default="./rag/feature_cache/bert")
    parser.add_argument("--artifact_dir", default="./rag/target_poi_multiview/artifacts_dropout04_wd3e3_dst_time")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--train_output", default=None)
    parser.add_argument("--test_output", default=None)
    parser.add_argument("--stats_output", default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--retrieval_pool_k", type=int, default=20)
    parser.add_argument("--train_multiview_k", type=int, default=8)
    parser.add_argument("--same_category_k", type=int, default=4)
    parser.add_argument("--near_k", type=int, default=4)
    parser.add_argument("--easy_k", type=int, default=4)
    parser.add_argument("--popular_pool_k", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--max_graph_nodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    data_dir = f"./datasets/{args.dataset_name}/preprocessed"
    tag = args.candidate_tag
    args.train_csv = args.train_csv or f"{data_dir}/train_sample.csv"
    args.test_csv = args.test_csv or f"{data_dir}/test_sample_with_traj.csv"
    args.train_qa = args.train_qa or f"{data_dir}/train_qa_pairs_kqt.json"
    args.test_qa = args.test_qa or f"{data_dir}/test_qa_pairs_kqt.txt"
    args.test_qa_for_dataset = args.test_qa_for_dataset or f"{data_dir}/test_qa_pairs_kqt.json"
    args.poi_info = args.poi_info or f"{data_dir}/poi_info.csv"
    args.poi_desc = args.poi_desc or f"{data_dir}/poi_desc_with_id.csv"
    args.train_output = args.train_output or (
        f"{data_dir}/train_qa_pairs_kqt_candidates_target_poi_multiview_{tag}.json"
    )
    args.test_output = args.test_output or (
        f"{data_dir}/test_qa_pairs_kqt_candidates_target_poi_multiview_dst_time_top20.txt"
    )
    args.stats_output = args.stats_output or f"{data_dir}/target_poi_multiview_{tag}_stats.json"

    train_rows = load_rows(args.train_qa)
    test_rows = load_rows(args.test_qa)
    poi_info = load_poi_info(args.poi_info, args.poi_desc, train_rows, test_rows)

    artifact_dir = Path(args.artifact_dir)
    model_path = args.model_path or str(artifact_dir / "model.pth")
    config_path = artifact_dir / "config.pth"
    if config_path.exists():
        cfg = torch.load(config_path, map_location="cpu")
        args.hidden_size = int(cfg.get("hidden_size", args.hidden_size))
        args.dropout = float(cfg.get("dropout", args.dropout))
        args.fusion_type = str(cfg.get("fusion_type", "gated"))
        args.fusion_heads = int(cfg.get("fusion_heads", 4))
        args.graph_alignment = str(cfg.get("graph_alignment", "none"))
        args.graph_align_heads = int(cfg.get("graph_align_heads", 4))
        args.graph_align_residual = bool(cfg.get("graph_align_residual", True))
        args.max_seq_len = int(cfg.get("max_seq_len", args.max_seq_len))
        args.max_graph_nodes = int(cfg.get("max_graph_nodes", args.max_graph_nodes))

    device = torch.device(args.device)
    tm = mv.TimeBucketManager()
    ds = mv.load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa_for_dataset)
    all_pois = sorted(int(x) for x in ds.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
    idx_to_poi = {i: pid for pid, i in p2i.items()}
    sem, geo = mv.load_feature_cache(args.feature_cache, all_pois)
    transition_index = mv.build_transition_index(ds, p2i, tm)
    graph_builder = mv.SubgraphBuilder(transition_index, max_nodes=args.max_graph_nodes)

    train_samples, train_raw_indices, train_skipped = build_export_samples(train_rows, p2i, tm, graph_builder, args.max_seq_len)
    test_samples, test_raw_indices, test_skipped = build_export_samples(test_rows, p2i, tm, graph_builder, args.max_seq_len)

    model = mv.MultiViewPOIRetriever(
        sem,
        geo,
        args.hidden_size,
        args.dropout,
        fusion_type=getattr(args, "fusion_type", "gated"),
        fusion_heads=getattr(args, "fusion_heads", 4),
        graph_alignment=getattr(args, "graph_alignment", "none"),
        graph_align_heads=getattr(args, "graph_align_heads", 4),
        graph_align_residual=getattr(args, "graph_align_residual", True),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    train_loader = DataLoader(mv.MultiViewDataset(train_samples), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mv.collate_fn)
    test_loader = DataLoader(mv.MultiViewDataset(test_samples), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mv.collate_fn)
    train_retrieval = retrieve_candidates(model, train_loader, idx_to_poi, device, args.retrieval_pool_k)
    test_retrieval = retrieve_candidates(model, test_loader, idx_to_poi, device, args.top_k)

    category_index = build_category_index(poi_info, all_pois)
    coords = build_near_index(poi_info, all_pois)
    target_counts = Counter(parse_target(row.get("answer", "")) for row in train_rows)
    popular_pois = [int(pid) for pid, _count in target_counts.most_common(args.popular_pool_k) if pid is not None]

    train_candidate_map = {}
    for raw_idx, cands in zip(train_raw_indices, train_retrieval):
        rng = random.Random(args.random_seed + raw_idx)
        train_candidate_map[raw_idx] = build_train_mixed_candidates(
            train_rows[raw_idx], cands, all_pois, poi_info, category_index, coords, popular_pois, rng, args
        )
    test_candidate_map = {
        raw_idx: build_test_topk_candidates(cands, args.top_k)
        for raw_idx, cands in zip(test_raw_indices, test_retrieval)
    }

    train_out, train_stats = build_augmented_rows(train_rows, train_candidate_map, poi_info)
    test_out, test_stats = build_augmented_rows(test_rows, test_candidate_map, poi_info)
    train_stats["retrieved_rows"] = len(train_candidate_map)
    train_stats["skipped_rows"] = len(train_skipped)
    test_stats["retrieved_rows"] = len(test_candidate_map)
    test_stats["skipped_rows"] = len(test_skipped)

    save_json(train_out, args.train_output)
    save_txt(test_out, args.test_output)
    stats = {
        "rule": "train: 8 random-sampled multiview top20 + 4 same-catego
