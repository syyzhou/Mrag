import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import export_layered_candidates as base  # noqa: E402
import train as mv  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def softmax(values, temperature=1.0):
    if not values:
        return []
    scaled = [float(v) / max(float(temperature), 1e-6) for v in values]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    z = sum(exps)
    return [v / max(z, 1e-12) for v in exps]


def normalized_entropy(probs):
    if len(probs) <= 1:
        return 0.0
    ent = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return ent / math.log(len(probs))


def weighted_sample_without_replacement(items, weights, k, rng):
    items = list(items)
    weights = [max(float(w), 0.0) for w in weights]
    out = []
    for _ in range(min(k, len(items))):
        total = sum(weights)
        if total <= 0:
            idx = rng.randrange(len(items))
        else:
            r = rng.random() * total
            acc = 0.0
            idx = len(items) - 1
            for i, w in enumerate(weights):
                acc += w
                if acc >= r:
                    idx = i
                    break
        out.append(items.pop(idx))
        weights.pop(idx)
    return out


def random_insert_preserve_order(head, supplements, top_k, rng):
    head = sorted(list(head), key=lambda x: int(x.get("retrieval_rank", 10**9)))
    supplements = sorted(list(supplements), key=lambda x: int(x.get("retrieval_rank", 10**9)))
    supplements = supplements[: max(0, top_k - len(head))]
    if not supplements:
        return head[:top_k]
    sup_positions = set(rng.sample(range(top_k), min(len(supplements), top_k)))
    out = []
    head_i = 0
    sup_i = 0
    for pos in range(top_k):
        if pos in sup_positions and sup_i < len(supplements):
            out.append(supplements[sup_i])
            sup_i += 1
        elif head_i < len(head):
            out.append(head[head_i])
            head_i += 1
        elif sup_i < len(supplements):
            out.append(supplements[sup_i])
            sup_i += 1
    while len(out) < top_k and head_i < len(head):
        out.append(head[head_i])
        head_i += 1
    while len(out) < top_k and sup_i < len(supplements):
        out.append(supplements[sup_i])
        sup_i += 1
    return out[:top_k]


def refill_to_top_k(selected, pool, args, target_poi=None):
    seen = {int(c["poi_id"]) for c in selected}
    for cand in pool:
        pid = int(cand["poi_id"])
        if pid in seen:
            continue
        if target_poi is not None and pid == int(target_poi):
            continue
        selected.append(dict(cand, source="uncertainty_refill"))
        seen.add(pid)
        if len(selected) >= args.top_k:
            break
    return selected


def split_args(args, train_mode):
    if train_mode:
        return {
            "retrieval_top_k": args.train_retrieval_top_k,
            "head_k": args.train_head_k,
            "tau_min": args.train_tau_min,
            "tau_max": args.train_tau_max,
        }
    return {
        "retrieval_top_k": args.test_retrieval_top_k,
        "head_k": args.test_head_k,
        "tau_min": args.test_tau_min,
        "tau_max": args.test_tau_max,
    }


def calibrate_target_coverage(selected, pool, target_poi, raw_idx, args, train_mode):
    if not train_mode or target_poi is None or args.train_target_coverage < 0:
        return selected

    target_poi = int(target_poi)
    target_pos = next((i for i, c in enumerate(selected) if int(c["poi_id"]) == target_poi), None)
    rng = random.Random(args.random_seed + 1000003 + int(raw_idx))

    if target_pos is not None:
        if rng.random() > args.train_target_coverage:
            selected.pop(target_pos)
            selected = refill_to_top_k(selected, pool, args, target_poi=target_poi)
    else:
        target_row = next((c for c in pool if int(c["poi_id"]) == target_poi), None)
        if target_row is not None and rng.random() < args.train_target_coverage:
            if len(selected) >= args.top_k:
                removable = [
                    i for i, c in enumerate(selected)
                    if int(c.get("retrieval_rank", args.train_retrieval_top_k + 1)) > args.train_head_k
                ]
                drop_idx = removable[-1] if removable else len(selected) - 1
                selected.pop(drop_idx)
            selected.append(dict(target_row, source="target_coverage_calibrated"))

    selected = selected[: args.top_k]
    selected = sorted(selected, key=lambda x: int(x.get("retrieval_rank", 10**9)))
    return selected


def build_uncertainty_candidates(retrieval_rows, raw_idx, args, train_mode, target_poi=None):
    cfg = split_args(args, train_mode)
    pool = retrieval_rows[: cfg["retrieval_top_k"]]
    if not pool:
        return [], {"entropy": 0.0, "tau": cfg["tau_min"]}

    scores = [float(c["score"]) for c in pool]
    base_probs = softmax(scores, temperature=1.0)
    entropy = normalized_entropy(base_probs)
    tau = cfg["tau_min"] + entropy * (cfg["tau_max"] - cfg["tau_min"])
    calibrated_probs = softmax(scores, temperature=tau)

    if train_mode:
        rng = random.Random(args.random_seed + int(raw_idx))
        head = [dict(c, source="uncertainty_head") for c in pool[: cfg["head_k"]]]
        head_ids = {int(c["poi_id"]) for c in head}
        tail_items = [c for c in pool[cfg["head_k"] :] if int(c["poi_id"]) not in head_ids]
        tail_probs = [calibrated_probs[pool.index(c)] for c in tail_items]
        sampled = weighted_sample_without_replacement(tail_items, tail_probs, args.top_k - len(head), rng)
        supplements = [dict(c, source="uncertainty_sample") for c in sampled]
        if args.train_random_insert:
            selected = random_insert_preserve_order(head, supplements, args.top_k, rng)
        else:
            selected = sorted(head + supplements, key=lambda x: int(x.get("retrieval_rank", 10**9)))
    else:
        rng = random.Random(args.random_seed + 2000003 + int(raw_idx))
        head = [dict(c, source="uncertainty_head") for c in pool[: cfg["head_k"]]]
        head_ids = {int(c["poi_id"]) for c in head}
        tail_items = [c for c in pool[cfg["head_k"] :] if int(c["poi_id"]) not in head_ids]
        if args.test_sample_supplements:
            tail_probs = [calibrated_probs[pool.index(c)] for c in tail_items]
            supplements = weighted_sample_without_replacement(tail_items, tail_probs, args.top_k - len(head), rng)
        else:
            supplements = tail_items[: args.top_k - len(head)]
        supplements = [dict(c, source="uncertainty_test_supplement") for c in supplements]
        if args.test_random_insert:
            selected = random_insert_preserve_order(head, supplements, args.top_k, rng)
        else:
            selected = sorted(head + supplements, key=lambda x: int(x.get("retrieval_rank", 10**9)))

    selected = calibrate_target_coverage(selected, pool, target_poi, raw_idx, args, train_mode)
    selected = refill_to_top_k(selected, pool, args)
    selected = selected[: args.top_k]
    for rank, cand in enumerate(selected, start=1):
        cand["rank"] = rank
        cand["uncertainty_entropy"] = float(entropy)
        cand["uncertainty_tau"] = float(tau)
    return selected, {"entropy": entropy, "tau": tau}


@torch.no_grad()
def retrieve_candidates(model, loader, idx_to_poi, device, top_k):
    model.eval()
    all_poi = model.poi_encoder.all_embeddings()
    rows = []
    for batch in base.tqdm(loader, desc="retrieve"):
        batch = mv.move_batch(batch, device)
        scores = model(batch)["fused"] @ all_poi.t()
        vals, inds = scores.topk(top_k, dim=-1)
        for b in range(inds.size(0)):
            rows.append(
                [
                    {
                        "poi_id": int(idx_to_poi[int(inds[b, j].item())]),
                        "score": float(vals[b, j].item()),
                        "retrieval_rank": int(j + 1),
                    }
                    for j in range(inds.size(1))
                ]
            )
    return rows


def build_candidate_map(raw_rows, raw_indices, retrieval_rows, args, train_mode):
    candidate_map = {}
    entropies = []
    taus = []
    for raw_idx, cands in zip(raw_indices, retrieval_rows):
        target_poi = base.parse_target(raw_rows[raw_idx].get("answer", ""))
        selected, meta = build_uncertainty_candidates(
            cands,
            raw_idx,
            args,
            train_mode=train_mode,
            target_poi=target_poi,
        )
        candidate_map[raw_idx] = selected
        entropies.append(meta["entropy"])
        taus.append(meta["tau"])
    return candidate_map, {
        "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "mean_tau": sum(taus) / len(taus) if taus else 0.0,
    }


def main():
    default_dataset = os.getenv("DATASET_NAME", "nyc")
    default_data_dir = f"./datasets/{default_dataset}/preprocessed"
    default_tag = "uncertainty_head10_pool100_tau07_20"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=default_dataset)
    parser.add_argument("--candidate_tag", default=default_tag)
    parser.add_argument("--train_csv", default=f"{default_data_dir}/train_sample.csv")
    parser.add_argument("--test_csv", default=f"{default_data_dir}/test_sample_with_traj.csv")
    parser.add_argument("--train_qa", default=f"{default_data_dir}/train_qa_pairs_kqt.json")
    parser.add_argument("--test_qa", default=f"{default_data_dir}/test_qa_pairs_kqt.txt")
    parser.add_argument("--test_qa_for_dataset", default=f"{default_data_dir}/test_qa_pairs_kqt.json")
    parser.add_argument("--poi_info", default=f"{default_data_dir}/poi_info.csv")
    parser.add_argument("--poi_desc", default=f"{default_data_dir}/poi_desc_with_id.csv")
    parser.add_argument("--feature_cache", default=f"./rag/feature_cache/bert_{default_dataset}")
    parser.add_argument("--artifact_dir", default=f"./rag/target_poi_multiview/artifacts_dropout04_wd3e3_{default_dataset}")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--train_output", default=None)
    parser.add_argument("--test_output", default=None)
    parser.add_argument("--stats_output", default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--retrieval_top_k", type=int, default=100)
    parser.add_argument("--head_k", type=int, default=10)
    parser.add_argument("--tau_min", type=float, default=0.7)
    parser.add_argument("--tau_max", type=float, default=2.0)
    parser.add_argument("--train_retrieval_top_k", type=int, default=None)
    parser.add_argument("--test_retrieval_top_k", type=int, default=None)
    parser.add_argument("--train_head_k", type=int, default=None)
    parser.add_argument("--test_head_k", type=int, default=None)
    parser.add_argument("--train_tau_min", type=float, default=None)
    parser.add_argument("--train_tau_max", type=float, default=None)
    parser.add_argument("--test_tau_min", type=float, default=None)
    parser.add_argument("--test_tau_max", type=float, default=None)
    parser.add_argument("--train_random_insert", type=str2bool, default=True)
    parser.add_argument("--test_random_insert", type=str2bool, default=True)
    parser.add_argument("--test_sample_supplements", type=str2bool, default=True)
    parser.add_argument("--train_target_coverage", type=float, default=-1.0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--max_graph_nodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.train_retrieval_top_k = args.train_retrieval_top_k or args.retrieval_top_k
    args.test_retrieval_top_k = args.test_retrieval_top_k or min(args.retrieval_top_k, args.top_k)
    args.train_head_k = args.train_head_k if args.train_head_k is not None else args.head_k
    args.test_head_k = args.test_head_k if args.test_head_k is not None else args.top_k
    args.train_tau_min = args.train_tau_min if args.train_tau_min is not None else args.tau_min
    args.train_tau_max = args.train_tau_max if args.train_tau_max is not None else args.tau_max
    args.test_tau_min = args.test_tau_min if args.test_tau_min is not None else args.tau_min
    args.test_tau_max = args.test_tau_max if args.test_tau_max is not None else args.tau_max

    data_dir = f"./datasets/{args.dataset_name}/preprocessed"
    args.train_output = args.train_output or (
        f"{data_dir}/train_qa_pairs_kqt_candidates_target_poi_multiview_{args.candidate_tag}.json"
    )
    args.test_output = args.test_output or (
        f"{data_dir}/test_qa_pairs_kqt_candidates_target_poi_multiview_{args.candidate_tag}.txt"
    )
    args.stats_output = args.stats_output or f"{data_dir}/target_poi_multiview_{args.candidate_tag}_stats.json"

    train_rows = base.load_rows(args.train_qa)
    test_rows = base.load_rows(args.test_qa)
    poi_info = base.load_poi_info(args.poi_info, args.poi_desc, train_rows, test_rows)

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
    else:
        args.fusion_type = "gated"
        args.fusion_heads = 4
        args.graph_alignment = "none"
        args.graph_align_heads = 4
        args.graph_align_residual = True

    device = torch.device(args.device)
    tm = mv.TimeBucketManager()
    ds = mv.load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa_for_dataset)
    all_pois = sorted(int(x) for x in ds.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
    idx_to_poi = {i: pid for pid, i in p2i.items()}
    sem, geo = mv.load_feature_cache(args.feature_cache, all_pois)
    transition_index = mv.build_transition_index(ds, p2i, tm)
    graph_builder = mv.SubgraphBuilder(transition_index, max_nodes=args.max_graph_nodes)

    train_samples, train_raw_indices, train_skipped = base.build_export_samples(
        train_rows, p2i, tm, graph_builder, args.max_seq_len
    )
    test_samples, test_raw_indices, test_skipped = base.build_export_samples(
        test_rows, p2i, tm, graph_builder, args.max_seq_len
    )

    model = mv.MultiViewPOIRetriever(
        sem,
        geo,
        args.hidden_size,
        args.dropout,
        fusion_type=args.fusion_type,
        fusion_heads=args.fusion_heads,
        graph_alignment=args.graph_alignment,
        graph_align_heads=args.graph_align_heads,
        graph_align_residual=args.graph_align_residual,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    train_loader = DataLoader(
        mv.MultiViewDataset(train_samples), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mv.collate_fn
    )
    test_loader = DataLoader(
        mv.MultiViewDataset(test_samples), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mv.collate_fn
    )
    train_retrieval = retrieve_candidates(model, train_loader, idx_to_poi, device, args.train_retrieval_top_k)
    test_retrieval = retrieve_candidates(model, test_loader, idx_to_poi, device, args.test_retrieval_top_k)

    train_candidate_map, train_dist_stats = build_candidate_map(
        train_rows, train_raw_indices, train_retrieval, args, train_mode=True
    )
    test_candidate_map, test_dist_stats = build_candidate_map(
        test_rows, test_raw_indices, test_retrieval, args, train_mode=False
    )

    train_out, train_stats = base.build_augmented_rows(train_rows, train_candidate_map, poi_info)
    test_out, test_stats = base.build_augmented_rows(test_rows, test_candidate_map, poi_info)
    train_stats.update(train_dist_stats)
    test_stats.update(test_dist_stats)
    train_stats["retrieved_rows"] = len(train_candidate_map)
    train_stats["skipped_rows"] = len(train_skipped)
    test_stats["retrieved_rows"] = len(test_candidate_map)
    test_stats["skipped_rows"] = len(test_skipped)

    base.save_json(train_out, args.train_output)
    base.save_txt(test_out, args.test_output)
    stats = {
        "rule": (
            "train: preserve retriever head_k and sample remaining candidates without replacement "
            "from an entropy-calibrated softmax over retrieval_top_k; test: deterministic retriever top_k"
        ),
        "model_path": model_path,
        "artifact_dir": str(artifact_dir),
        "random_seed": args.random_seed,
        "retrieval_top_k": args.retrieval_top_k,
        "train_retrieval_top_k": args.train_retrieval_top_k,
        "test_retrieval_top_k": args.test_retrieval_top_k,
        "top_k": args.top_k,
        "train_head_k": args.train_head_k,
        "test_head_k": args.test_head_k,
        "train_tau_min": args.train_tau_min,
        "train_tau_max": args.train_tau_max,
        "test_tau_min": args.test_tau_min,
        "test_tau_max": args.test_tau_max,
        "train_random_insert": args.train_random_insert,
        "test_random_insert": args.test_random_insert,
        "test_sample_supplements": args.test_sample_supplements,
        "train_target_coverage": args.train_target_coverage,
        "train": train_stats,
        "test": test_stats,
        "outputs": {"train": args.train_output, "test": args.test_output},
    }
    base.save_json(stats, args.stats_output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
