import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import (
    MultiViewDataset,
    MultiViewPOIRetriever,
    SubgraphBuilder,
    TimeBucketManager,
    build_samples,
    build_transition_index,
    collate_fn,
    load_feature_cache,
    load_full_dataset,
    move_batch,
)


@torch.no_grad()
def evaluate_rank_metrics(model, samples, batch_size, device):
    loader = DataLoader(
        MultiViewDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    model.eval()
    all_poi = model.poi_encoder.all_embeddings()

    total = 0
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    rr_sum = 0.0
    rr20_sum = 0.0
    rank_sum = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        scores = model(batch)["fused"] @ all_poi.t()
        target = batch["target"]
        target_scores = scores.gather(1, target.view(-1, 1))
        ranks = (scores > target_scores).sum(dim=1) + 1

        total += target.numel()
        ranks_f = ranks.float()
        rr = 1.0 / ranks_f
        rr_sum += rr.sum().item()
        rr20_sum += torch.where(ranks <= 20, rr, torch.zeros_like(rr)).sum().item()
        rank_sum += ranks_f.sum().item()

        for k in hits:
            hits[k] += (ranks <= k).sum().item()

    metrics = {
        "total": total,
        "mrr_full": rr_sum / max(total, 1),
        "mrr_at_20": rr20_sum / max(total, 1),
        "mean_rank": rank_sum / max(total, 1),
    }
    for k in hits:
        metrics[f"recall@{k}"] = hits[k] / max(total, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default="./rag/target_poi_multiview/artifacts_dropout04_wd3e3_dst_time")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    model_path = Path(args.model_path) if args.model_path else artifact_dir / "model.pth"
    config = torch.load(artifact_dir / "config.pth", map_location="cpu")
    cfg = Namespace(**config)

    device = torch.device(args.device)
    batch_size = args.batch_size or cfg.eval_batch_size

    tm = TimeBucketManager()
    dataset = load_full_dataset(cfg.train_csv, cfg.test_csv, cfg.train_qa, cfg.test_qa)
    all_pois = sorted(int(x) for x in dataset.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
    sem_vectors, geo_features = load_feature_cache(cfg.feature_cache, all_pois)

    transition_index = build_transition_index(dataset, p2i, tm)
    graph_builder = SubgraphBuilder(transition_index, max_nodes=cfg.max_graph_nodes)
    train_samples = build_samples(cfg.train_qa, p2i, tm, graph_builder, cfg.max_seq_len, cfg.max_train_samples)
    test_samples = build_samples(cfg.test_qa, p2i, tm, graph_builder, cfg.max_seq_len, cfg.max_val_samples)

    model = MultiViewPOIRetriever(
        sem_vectors,
        geo_features,
        cfg.hidden_size,
        cfg.dropout,
        fusion_type=getattr(cfg, "fusion_type", "gated"),
        fusion_heads=getattr(cfg, "fusion_heads", 4),
        graph_alignment=getattr(cfg, "graph_alignment", "none"),
        graph_align_heads=getattr(cfg, "graph_align_heads", 4),
        graph_align_residual=getattr(cfg, "graph_align_residual", True),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    result = {
        "artifact_dir": str(artifact_dir),
        "model_path": str(model_path),
        "device": str(device),
        "batch_size": batch_size,
        "train": evaluate_rank_metrics(model, train_samples, batch_size, device),
        "test": evaluate_rank_metrics(model, test_samples, batch_size, device),
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
