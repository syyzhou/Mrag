import argparse
import json
import os
from datetime import datetime

import numpy as np

from retrieval_utils import load_full_dataset, prepare_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build reusable frozen POI text/geo feature cache for rag retrievers."
    )
    parser.add_argument("--train_csv", default="./datasets/nyc/preprocessed/train_sample.csv")
    parser.add_argument("--test_csv", default="./datasets/nyc/preprocessed/test_sample.csv")
    parser.add_argument("--train_qa", default="./datasets/nyc/preprocessed/train_qa_pairs_kqt.json")
    parser.add_argument("--test_qa", default="./datasets/nyc/preprocessed/test_qa_pairs_kqt.json")
    parser.add_argument("--encoder", default="bert")
    parser.add_argument("--output_dir", default="./rag/feature_cache/bert")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sem_path = os.path.join(args.output_dir, "poi_sem_vectors.npy")
    geo_path = os.path.join(args.output_dir, "poi_geo_features.npy")
    poi_path = os.path.join(args.output_dir, "poi_id_list.json")
    meta_path = os.path.join(args.output_dir, "feature_meta.json")

    existing = [sem_path, geo_path, poi_path, meta_path]
    if not args.force and any(os.path.exists(path) for path in existing):
        missing = [path for path in existing if not os.path.exists(path)]
        if missing:
            raise RuntimeError(
                "Feature cache is incomplete. Re-run with --force after checking the directory: "
                + ", ".join(missing)
            )
        print(f"Feature cache already exists at {args.output_dir}. Use --force to rebuild.")
        return

    dataset = load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa)
    all_pois = sorted(dataset.poi_dict.keys())

    features = prepare_features(
        dataset=dataset,
        tg=None,
        all_pois=all_pois,
        encoder_name=args.encoder,
        device=args.device,
        use_llm=True,
    )

    sem_vectors = np.asarray(features["sem_vectors"], dtype=np.float32)
    geo_features = np.asarray(features["geo_features"], dtype=np.float32)

    if sem_vectors.shape[0] != len(all_pois):
        raise RuntimeError(
            f"sem_vectors rows {sem_vectors.shape[0]} != number of POIs {len(all_pois)}"
        )
    if geo_features.shape[0] != len(all_pois):
        raise RuntimeError(
            f"geo_features rows {geo_features.shape[0]} != number of POIs {len(all_pois)}"
        )

    np.save(sem_path, sem_vectors)
    np.save(geo_path, geo_features)
    with open(poi_path, "w", encoding="utf-8") as f:
        json.dump(all_pois, f, ensure_ascii=False)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "encoder": args.encoder,
        "device": args.device,
        "num_pois": len(all_pois),
        "llm_dim": int(features["llm_dim"]),
        "sem_vectors_shape": list(sem_vectors.shape),
        "geo_features_shape": list(geo_features.shape),
        "poi_id_order": "sorted(dataset.poi_dict.keys())",
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "train_qa": args.train_qa,
        "test_qa": args.test_qa,
        "geo_norm": features.get("geo_norm", {}),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved POI feature cache to {args.output_dir}")
    print(f"  sem_vectors: {sem_vectors.shape} -> {sem_path}")
    print(f"  geo_features: {geo_features.shape} -> {geo_path}")
    print(f"  poi_ids: {len(all_pois)} -> {poi_path}")
    print(f"  meta -> {meta_path}")


if __name__ == "__main__":
    main()
