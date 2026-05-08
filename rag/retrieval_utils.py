import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import pygeohash as pgh
    HAS_GEOHASH = True
except ImportError:
    HAS_GEOHASH = False

try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class POIInfo:
    poi_id: int
    category_id: int
    category_name: str
    latitude: float
    longitude: float
    visit_count: int = 0

    def to_text(self) -> str:
        return f"{self.category_name} (POI {self.poi_id})"


@dataclass
class Visit:
    poi_id: int
    timestamp: str
    epoch: int
    hour: int
    day_of_week: int
    is_weekend: bool
    latitude: float
    longitude: float
    category_id: int
    category_name: str
    user_id: int
    trajectory_id: str


@dataclass
class Trajectory:
    trajectory_id: str
    user_id: int
    visits: List[Visit]

    def get_poi_sequence(self) -> List[int]:
        return [v.poi_id for v in self.visits]

    def get_last_visit(self) -> Visit:
        return self.visits[-1]

    def get_context_window(self, window_size: int = 5) -> List[Visit]:
        return self.visits[-window_size:]


@dataclass
class QAPair:
    question: str
    answer: str
    target_poi_id: int
    target_poi_category: str
    target_time: str
    user_id: int
    current_traj_visits: List[Dict]
    historical_sequences: List[List[Dict]]


@dataclass
class POIDatasetFull:
    poi_dict: Dict[int, POIInfo]
    all_trajectories: Dict[str, Trajectory]
    user_trajectories: Dict[int, List[str]]
    train_qa_pairs: List[QAPair]
    test_qa_pairs: List[QAPair]
    num_pois: int
    num_users: int
    num_categories: int
    category_id_to_name: Dict[int, str]
    category_name_to_id: Dict[str, int]


def parse_timestamp(ts_str: str) -> Tuple[int, int, bool]:
    dt = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
    return dt.hour, dt.weekday(), dt.weekday() >= 5


def load_csv_data(csv_path: str) -> Tuple[Dict[int, POIInfo], Dict[str, Trajectory], Dict[int, List[str]]]:
    print(f"  Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "trajectory_id" in df.columns:
        traj_col = "trajectory_id"
    elif "pseudo_session_trajectory_id" in df.columns:
        traj_col = "pseudo_session_trajectory_id"
    else:
        raise KeyError(
            f"Missing trajectory column in {csv_path}. Expected one of: "
            f"trajectory_id, pseudo_session_trajectory_id. Got columns: {list(df.columns)}"
        )

    poi_dict = {}
    df = df.sort_values("UTCTimeOffsetEpoch")

    for _, row in df.iterrows():
        pid = int(row["PoiId"])
        if pid not in poi_dict:
            poi_dict[pid] = POIInfo(
                poi_id=pid,
                category_id=int(row["PoiCategoryId"]),
                category_name=str(row["PoiCategoryName"]).strip(),
                latitude=float(row["Latitude"]),
                longitude=float(row["Longitude"]),
                visit_count=0,
            )
        poi_dict[pid].visit_count += 1

    trajectories = {}
    user_trajectories = defaultdict(list)
    for traj_id, group in df.groupby(traj_col):
        traj_id = str(traj_id)
        group = group.sort_values("UTCTimeOffsetEpoch")
        user_id = int(group.iloc[0]["UserId"])
        visits = []
        for _, row in group.iterrows():
            hour, dow, is_wknd = parse_timestamp(str(row["UTCTimeOffset"]))
            visits.append(
                Visit(
                    poi_id=int(row["PoiId"]),
                    timestamp=str(row["UTCTimeOffset"]).strip(),
                    epoch=int(row["UTCTimeOffsetEpoch"]),
                    hour=hour,
                    day_of_week=dow,
                    is_weekend=is_wknd,
                    latitude=float(row["Latitude"]),
                    longitude=float(row["Longitude"]),
                    category_id=int(row["PoiCategoryId"]),
                    category_name=str(row["PoiCategoryName"]).strip(),
                    user_id=user_id,
                    trajectory_id=traj_id,
                )
            )
        if len(visits) >= 2:
            trajectories[traj_id] = Trajectory(traj_id, user_id, visits)
            user_trajectories[user_id].append(traj_id)

    print(f"    POIs: {len(poi_dict)}, Trajectories: {len(trajectories)}, Users: {len(user_trajectories)}")
    return poi_dict, trajectories, user_trajectories


def _parse_qa_fields(question: str, answer: str) -> QAPair:
    target_poi_id, target_category, target_time, user_id = -1, "", "", -1
    ans_match = re.search(r"At (.+?), user (\d+) will visit POI id (\d+)\.(.+?)\.", answer)
    if ans_match:
        target_time = ans_match.group(1).strip()
        target_poi_id = int(ans_match.group(3))
        target_category = ans_match.group(4).strip()

    user_match = re.search(r"check-in sequences of user (\d+)", question)
    if user_match:
        user_id = int(user_match.group(1))

    current_visits = []
    current_section = re.search(
        r"\[Current trajectory's check-in sequence\]:(.*?)(?:\[Historical|Given the data)",
        question,
        re.DOTALL,
    )
    if current_section:
        for m in re.finditer(
            r"At (.+?), user \d+ visited POI id (\d+) which is a (.+?) with Category id (\d+)",
            current_section.group(1),
        ):
            current_visits.append(
                {
                    "timestamp": m.group(1).strip(),
                    "poi_id": int(m.group(2)),
                    "category_name": m.group(3).strip(),
                    "category_id": int(m.group(4)),
                }
            )

    historical_sequences = []
    for hs in re.finditer(
        r"\[Sequence from .+?\]:(.*?)(?=\[Sequence from|\[Historical|Given the data)",
        question,
        re.DOTALL,
    ):
        seq = []
        for m in re.finditer(
            r"At (.+?), user \d+ visited POI id (\d+) which is a (.+?) with Category id (\d+)",
            hs.group(1),
        ):
            seq.append(
                {
                    "timestamp": m.group(1).strip(),
                    "poi_id": int(m.group(2)),
                    "category_name": m.group(3).strip(),
                    "category_id": int(m.group(4)),
                }
            )
        if seq:
            historical_sequences.append(seq)

    return QAPair(
        question=question,
        answer=answer,
        target_poi_id=target_poi_id,
        target_poi_category=target_category,
        target_time=target_time,
        user_id=user_id,
        current_traj_visits=current_visits,
        historical_sequences=historical_sequences,
    )


def parse_qa_json(json_path: str) -> List[QAPair]:
    print(f"  Loading QA JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    qa_pairs = [_parse_qa_fields(item["question"], item["answer"]) for item in raw_data]
    print(f"    Parsed {len(qa_pairs)} QA pairs")
    return qa_pairs


def parse_qa_txt(txt_path: str) -> List[QAPair]:
    print(f"  Loading QA TXT: {txt_path}")
    qa_pairs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "<answer>:" not in line:
                continue
            q_part, a_part = line.split("<answer>:", 1)
            question = q_part.replace("<question>:", "").strip()
            qa_pairs.append(_parse_qa_fields(question, a_part.strip()))
    print(f"    Parsed {len(qa_pairs)} QA pairs from TXT")
    return qa_pairs


def load_full_dataset(train_csv: str, test_csv: str, train_qa_json: str, test_qa_json: str) -> POIDatasetFull:
    print("=" * 60)
    print("Loading full dataset...")
    print("=" * 60)

    train_pois, train_trajs, train_ut = load_csv_data(train_csv)
    _test_pois, _test_trajs, test_ut = load_csv_data(test_csv)

    poi_dict = {**train_pois}
    all_trajs = train_trajs
    user_trajs = defaultdict(list)
    for uid, tids in train_ut.items():
        user_trajs[uid].extend(tids)
    for uid, tids in test_ut.items():
        user_trajs[uid].extend(tids)

    def load_qa_auto(path: str) -> List[QAPair]:
        if path.endswith(".json"):
            return parse_qa_json(path)
        if path.endswith(".txt"):
            return parse_qa_txt(path)
        raise ValueError(f"Unsupported QA format: {path}")

    train_qa = load_qa_auto(train_qa_json)
    test_qa = load_qa_auto(test_qa_json)

    cat_id2name = {}
    cat_name2id = {}
    for poi in poi_dict.values():
        cat_id2name[poi.category_id] = poi.category_name
        cat_name2id[poi.category_name] = poi.category_id

    num_pois = max(poi_dict.keys()) + 1 if poi_dict else 0
    num_users = max(user_trajs.keys()) + 1 if user_trajs else 0
    ds = POIDatasetFull(
        poi_dict=poi_dict,
        all_trajectories=all_trajs,
        user_trajectories=dict(user_trajs),
        train_qa_pairs=train_qa,
        test_qa_pairs=test_qa,
        num_pois=num_pois,
        num_users=num_users,
        num_categories=len(cat_id2name),
        category_id_to_name=cat_id2name,
        category_name_to_id=cat_name2id,
    )

    print(f"\n  Total POIs: {len(poi_dict)} (range 0~{num_pois - 1})")
    print(f"  Trajectories: {len(all_trajs)}, Users: {len(user_trajs)}")
    print(f"  Categories: {len(cat_id2name)}")
    print(f"  Train QA: {len(train_qa)}, Test QA: {len(test_qa)}")
    return ds


class TimeBucketManager:
    NUM_BUCKETS = 8
    PERIOD_NAMES = ["morning(06-11)", "noon(11-14)", "afternoon(14-18)", "evening(18-06)"]

    def __init__(self):
        self.num_buckets = self.NUM_BUCKETS

    def get_period(self, hour: int) -> int:
        if 6 <= hour < 11:
            return 0
        if 11 <= hour < 14:
            return 1
        if 14 <= hour < 18:
            return 2
        return 3

    def get_bucket(self, hour: int, is_weekend: bool) -> int:
        period = self.get_period(hour)
        return period + 4 if is_weekend else period

    def get_bucket_from_visit(self, visit) -> int:
        if isinstance(visit, dict):
            hour, _dow, is_weekend = parse_timestamp(visit["timestamp"])
            return self.get_bucket(hour, is_weekend)
        return self.get_bucket(visit.hour, visit.is_weekend)

    def get_bucket_name(self, bucket_id: int) -> str:
        day = "weekend" if bucket_id >= 4 else "weekday"
        return f"{day}_{self.PERIOD_NAMES[bucket_id % 4]}"

    def get_adjacent_buckets(self, bucket_id: int) -> List[int]:
        day_off = 4 if bucket_id >= 4 else 0
        period = bucket_id % 4
        adj = []
        if period > 0:
            adj.append(day_off + period - 1)
        if period < 3:
            adj.append(day_off + period + 1)
        adj.append((day_off + 4) % 8 + period)
        return adj


def build_transition_graph(dataset: POIDatasetFull, time_manager: TimeBucketManager) -> Dict:
    num_pois = dataset.num_pois
    num_buckets = time_manager.num_buckets
    print(f"  Building transition graph (pois={num_pois}, buckets={num_buckets})...")

    global_counts = np.zeros((num_pois, num_pois), dtype=np.float32)
    bucket_counts = {b: np.zeros((num_pois, num_pois), dtype=np.float32) for b in range(num_buckets)}
    cat_transitions = defaultdict(lambda: defaultdict(int))

    total_trans = 0
    for traj in dataset.all_trajectories.values():
        for i in range(len(traj.visits) - 1):
            src, dst = traj.visits[i], traj.visits[i + 1]
            s, d = src.poi_id, dst.poi_id
            if s >= num_pois or d >= num_pois:
                continue
            bucket = time_manager.get_bucket_from_visit(src)
            global_counts[s][d] += 1
            bucket_counts[bucket][s][d] += 1
            cat_transitions[src.category_name][dst.category_name] += 1
            total_trans += 1

    bucket_adj = {}
    for b in range(num_buckets):
        adj = {}
        mat = bucket_counts[b]
        for i in range(num_pois):
            row = mat[i]
            nz = np.nonzero(row)[0]
            if len(nz) > 0:
                si = nz[np.argsort(row[nz])[::-1]]
                adj[i] = [(int(j), float(row[j])) for j in si]
            else:
                adj[i] = []
        bucket_adj[b] = adj

    cat_transition_probs = {}
    for src_cat, dst_counts in cat_transitions.items():
        total = sum(dst_counts.values())
        cat_transition_probs[src_cat] = {
            dst_cat: count / total
            for dst_cat, count in sorted(dst_counts.items(), key=lambda x: -x[1])
        }

    srcs, dsts, ws = [], [], []
    for i in range(num_pois):
        for j in range(num_pois):
            if global_counts[i][j] > 0:
                srcs.append(i)
                dsts.append(j)
                ws.append(global_counts[i][j])

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long) if srcs else torch.zeros(2, 0, dtype=torch.long)
    edge_weight = torch.tensor(ws, dtype=torch.float) if ws else torch.zeros(0)
    print(f"    Total transitions: {total_trans}, Global edges: {len(srcs)}")
    return {
        "global_counts": global_counts,
        "bucket_counts": bucket_counts,
        "bucket_adj": bucket_adj,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_pois": num_pois,
        "num_buckets": num_buckets,
        "cat_transition_probs": cat_transition_probs,
    }


def get_top_transitions(tg: Dict, poi_id: int, bucket_id: int, top_k: int = 10, poi_dict: Dict = None) -> List[Dict]:
    adj = tg["bucket_adj"].get(bucket_id, {})
    neighbors = adj.get(poi_id, [])
    total = sum(w for _, w in neighbors) if neighbors else 0
    results = []
    for dst, w in neighbors[:top_k]:
        prob = w / total if total > 0 else 0
        row = {"poi_id": dst, "weight": w, "probability": prob}
        if poi_dict and dst in poi_dict:
            row["category"] = poi_dict[dst].category_name
            row["category_id"] = poi_dict[dst].category_id
        results.append(row)
    return results


class POITextEncoder:
    def __init__(self, encoder_name: str = "bert", device: str = "cpu"):
        self.encoder_name = encoder_name.lower()
        self.device = torch.device(device)
        model_map = {
            "bert": "RAG-GFM/bert-base-uncased",
            "e5": "intfloat/e5-base-v2",
            "bge": "BAAI/bge-base-en-v1.5",
        }
        target = model_map.get(self.encoder_name, "bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(target)
        self.textmodel = AutoModel.from_pretrained(target).to(self.device)
        self.textmodel.eval()
        for p in self.textmodel.parameters():
            p.requires_grad = False
        self.hidden_dim = self.textmodel.config.hidden_size

    def generate_prompt(self, poi: POIInfo, nearest_pois: List[Dict]) -> str:
        name = getattr(poi, "name", f"POI_{poi.poi_id}")
        prompt = f'The name of this location is "{name}". '
        prompt += f"Its POI category is {poi.category_name}"
        if hasattr(poi, "parent_category_name") and poi.parent_category_name:
            prompt += f", belonging to the parent category {poi.parent_category_name}"
        prompt += ". "
        prompt += (
            f"The geographic coordinates for this location are "
            f"({poi.latitude:.6f}, {poi.longitude:.6f})"
        )
        if HAS_GEOHASH:
            try:
                prompt += f", with the corresponding geohash code {pgh.encode(poi.latitude, poi.longitude, precision=7)}"
            except Exception:
                pass
        prompt += ". "
        if hasattr(poi, "address") and poi.address:
            prompt += f"The address is {poi.address}. "
        if "e5" in self.encoder_name:
            prompt = f"passage: {prompt}"
        return prompt

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=4096,
            ).to(self.device)
            out = self.textmodel(**inputs)
            emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)


def prepare_features(
    dataset: POIDatasetFull,
    tg: Dict,
    all_pois: List[int],
    encoder_name: str = "bert",
    device: str = "cpu",
    use_llm: bool = True,
) -> Dict:
    N = len(all_pois)
    lats = np.array([dataset.poi_dict[pid].latitude for pid in all_pois])
    lons = np.array([dataset.poi_dict[pid].longitude for pid in all_pois])
    R = 6378137.0
    mx = R * np.radians(lons)
    my = R * np.log(np.tan(np.pi / 4 + np.radians(lats) / 2))
    mx_m, mx_s = mx.mean(), mx.std() + 1e-6
    my_m, my_s = my.mean(), my.std() + 1e-6

    geo_features = np.zeros((N, 6), dtype=np.float32)
    for i in range(N):
        geo_features[i] = [
            (mx[i] - mx_m) / mx_s,
            (my[i] - my_m) / my_s,
            np.sin(np.radians(lats[i])),
            np.cos(np.radians(lats[i])),
            np.sin(np.radians(lons[i])),
            np.cos(np.radians(lons[i])),
        ]

    llm_dim = 768
    if use_llm and HAS_TRANSFORMERS:
        print(f"  Encoding POI texts with {encoder_name} (frozen)...")
        text_enc = POITextEncoder(encoder_name, device)
        llm_dim = text_enc.hidden_dim
        if HAS_SKLEARN:
            coords_rad = np.column_stack([np.radians(lats), np.radians(lons)])
            tree = BallTree(coords_rad, metric="haversine")
            dists_nn, idxs_nn = tree.query(coords_rad, k=6)
        else:
            idxs_nn = np.tile(np.arange(N).reshape(-1, 1), (1, 6))
            dists_nn = np.ones((N, 6))

        prompts = []
        for i, pid in enumerate(tqdm(all_pois, desc="  Generating prompts")):
            poi = dataset.poi_dict[pid]
            nearby = []
            for j in range(1, min(6, idxs_nn.shape[1])):
                nid = all_pois[idxs_nn[i][j]]
                npoi = dataset.poi_dict[nid]
                nearby.append(
                    {
                        "name": getattr(npoi, "name", f"POI_{nid}"),
                        "category": npoi.category_name,
                        "dist": float(dists_nn[i][j] * 6371),
                    }
                )
            prompts.append(text_enc.generate_prompt(poi, nearby))
        sem_vectors = text_enc.encode(prompts)
        print(f"  LLM vectors: {sem_vectors.shape}")
    else:
        print("  No LLM, using stat features as proxy")
        llm_dim = 32
        sem_vectors = np.zeros((N, llm_dim), dtype=np.float32)
        gc = tg["global_counts"]
        for i, pid in enumerate(all_pois):
            poi = dataset.poi_dict[pid]
            ch = hash(poi.category_name) % 256
            for j in range(8):
                sem_vectors[i, j] = float((ch >> j) & 1)
            sem_vectors[i, 8] = np.sin(poi.category_id * 0.1)
            sem_vectors[i, 9] = np.cos(poi.category_id * 0.1)
            sem_vectors[i, 10] = np.log1p(poi.visit_count) / 10
            if pid < gc.shape[0]:
                sem_vectors[i, 11] = np.log1p(gc[pid].sum()) / 10
                sem_vectors[i, 12] = np.log1p(gc[:, pid].sum()) / 10

    return {
        "sem_vectors": sem_vectors,
        "geo_features": geo_features,
        "llm_dim": llm_dim,
        "geo_norm": {
            "mx_m": float(mx_m),
            "mx_s": float(mx_s),
            "my_m": float(my_m),
            "my_s": float(my_s),
        },
    }
