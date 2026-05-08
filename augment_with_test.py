# augment_prompts_with_graph.py

import json
import re
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# 时间分桶 (8桶)
# ============================================================

PERIOD_NAMES = ['morning(06-11)', 'noon(11-14)', 'afternoon(14-18)', 'evening(18-06)']


def get_period(hour: int) -> int:
    if 6 <= hour < 11:
        return 0
    elif 11 <= hour < 14:
        return 1
    elif 14 <= hour < 18:
        return 2
    else:
        return 3


def get_bucket(dt: datetime) -> int:
    period = get_period(dt.hour)
    if dt.weekday() >= 5:
        return period + 4
    return period


def get_bucket_name(bucket_id: int) -> str:
    day_type = "weekend" if bucket_id >= 4 else "weekday"
    return f"{day_type}_{PERIOD_NAMES[bucket_id % 4]}"


def get_adjacent_buckets(bucket_id: int) -> List[int]:
    day_offset = 4 if bucket_id >= 4 else 0
    period = bucket_id % 4
    adj = []
    if period > 0:
        adj.append(day_offset + period - 1)
    if period < 3:
        adj.append(day_offset + period + 1)
    adj.append(((day_offset + 4) % 8) + period)
    return adj


# ============================================================
# 文本解析
# ============================================================

def parse_checkins_from_question(question: str) -> List[Dict]:
    pattern = (
        r'At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), '
        r'user (\d+) visited POI id (\d+) '
        r'which is a (.+?) with Category id (\d+)'
    )
    checkins = []
    for m in re.finditer(pattern, question):
        time_str = m.group(1)
        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        checkins.append({
            'time_str': time_str,
            'datetime': dt,
            'user_id': int(m.group(2)),
            'poi_id': int(m.group(3)),
            'category': m.group(4),
            'cat_id': int(m.group(5)),
        })
    checkins.sort(key=lambda x: x['datetime'])
    return checkins


def parse_target_from_question(question: str) -> Tuple[Optional[str], Optional[int]]:
    pattern = r'At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), Which POI id will user (\d+) visit'
    m = re.search(pattern, question)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def parse_answer(answer: str) -> Tuple[Optional[int], Optional[str]]:
    pattern = r'will visit POI id (\d+)\.(.+?)\.?$'
    m = re.search(pattern, answer.strip())
    if m:
        return int(m.group(1)), m.group(2).strip().rstrip('.')
    m2 = re.search(r'POI id (\d+)', answer)
    if m2:
        return int(m2.group(1)), ""
    return None, None


def parse_user_id_from_question(question: str) -> Optional[int]:
    m = re.search(r'user (\d+)', question)
    return int(m.group(1)) if m else None


# ============================================================
# 全局转移图
# ============================================================

class IncrementalTransitionGraph:
    def __init__(self):
        self.adj = defaultdict(lambda: defaultdict(lambda: {
            'weight': 0, 'category': '', 'cat_id': -1
        }))
        self.poi_info = {}
        self.bucket_transitions = defaultdict(list)
        self.total_edges = 0

    def add_transition(self, src_checkin: Dict, dst_checkin: Dict):
        src = src_checkin['poi_id']
        dst = dst_checkin['poi_id']

        self.poi_info[src] = {
            'category': src_checkin['category'],
            'cat_id': src_checkin.get('cat_id', -1)
        }
        self.poi_info[dst] = {
            'category': dst_checkin['category'],
            'cat_id': dst_checkin.get('cat_id', -1)
        }

        self.adj[src][dst]['weight'] += 1
        self.adj[src][dst]['category'] = dst_checkin['category']
        self.adj[src][dst]['cat_id'] = dst_checkin.get('cat_id', -1)

        bucket = get_bucket(dst_checkin['datetime'])
        self.bucket_transitions[bucket].append({
            'src_poi': src,
            'src_cat': src_checkin['category'],
            'src_cat_id': src_checkin.get('cat_id', -1),
            'dst_poi': dst,
            'dst_cat': dst_checkin['category'],
            'dst_cat_id': dst_checkin.get('cat_id', -1),
            'dst_time': dst_checkin['time_str'],
            'dst_dt': dst_checkin['datetime'],
            'bucket': bucket,
        })
        self.total_edges += 1

    def add_checkins_sequence(self, checkins: List[Dict]):
        for i in range(len(checkins) - 1):
            self.add_transition(checkins[i], checkins[i + 1])

    def get_adj(self) -> Dict:
        return dict(self.adj)

    def get_poi_info(self) -> Dict:
        return dict(self.poi_info)

    def get_triplets_for_bucket(self, bucket_id: int,
                                true_poi: int,
                                max_triplets: int = 20) -> List[Dict]:
        def contains_true(t):
            return t['src_poi'] == true_poi or t['dst_poi'] == true_poi

        matching = [t for t in self.bucket_transitions[bucket_id] if contains_true(t)]

        if len(matching) < max_triplets:
            for adj_b in get_adjacent_buckets(bucket_id):
                for t in self.bucket_transitions[adj_b]:
                    if contains_true(t) and t not in matching:
                        matching.append(t)
                        if len(matching) >= max_triplets:
                            break
                if len(matching) >= max_triplets:
                    break

        if len(matching) < max_triplets:
            for b, trans_list in self.bucket_transitions.items():
                for t in trans_list:
                    if contains_true(t) and t not in matching:
                        matching.append(t)
                        if len(matching) >= max_triplets:
                            break
                if len(matching) >= max_triplets:
                    break

        matching.sort(key=lambda x: x['dst_dt'], reverse=True)
        return matching[:max_triplets]


# ============================================================
# 子图提取
# ============================================================

def extract_subgraphs(adj, poi_info, true_poi, user_id, num_subgraphs=2):
    in_adj = defaultdict(lambda: defaultdict(int))
    for src, dsts in adj.items():
        for dst, info in dsts.items():
            in_adj[dst][src] += info['weight']

    node_out_weight = {}
    for node, neighbors in adj.items():
        node_out_weight[node] = sum(info['weight'] for info in neighbors.values())
    for node in in_adj:
        if node not in node_out_weight:
            node_out_weight[node] = 0

    sorted_nodes = sorted(node_out_weight.keys(),
                          key=lambda x: node_out_weight[x], reverse=True)

    subgraphs = []
    used = set()

    for center in sorted_nodes:
        if len(subgraphs) >= num_subgraphs:
            break
        if center in used:
            continue
        out_n = set(adj.get(center, {}).keys())
        in_n = set(in_adj.get(center, {}).keys())
        all_n = out_n | in_n
        if center != true_poi and true_poi not in all_n:
            continue
        subgraphs.append(_build_sg(center, adj, in_adj, poi_info, node_out_weight))
        used.add(center)

    if len(subgraphs) < num_subgraphs:
        for center in sorted_nodes:
            if len(subgraphs) >= num_subgraphs:
                break
            if center in used:
                continue
            out_n = set(adj.get(center, {}).keys())
            in_n = set(in_adj.get(center, {}).keys())
            one_hop = out_n | in_n
            found = False
            for n1 in one_hop:
                n1_all = set(adj.get(n1, {}).keys()) | set(in_adj.get(n1, {}).keys())
                if true_poi in n1_all:
                    found = True
                    break
            if not found:
                continue
            subgraphs.append(_build_sg(center, adj, in_adj, poi_info, node_out_weight))
            used.add(center)

    return subgraphs[:num_subgraphs]


def _build_sg(center, adj, in_adj, poi_info, node_out_weight):
    merged = defaultdict(lambda: {'out_weight': 0, 'in_weight': 0})
    for dst, info in adj.get(center, {}).items():
        merged[dst]['out_weight'] += info['weight']
    for src, w in in_adj.get(center, {}).items():
        merged[src]['in_weight'] += w

    neighbors = []
    for pid, wt in merged.items():
        info = poi_info.get(pid, {'category': 'Unknown', 'cat_id': -1})
        neighbors.append({
            'poi_id': pid,
            'total_weight': wt['out_weight'] + wt['in_weight'],
            'out_weight': wt['out_weight'],
            'in_weight': wt['in_weight'],
            'category': info['category'],
            'cat_id': info['cat_id'],
        })
    neighbors.sort(key=lambda x: x['total_weight'], reverse=True)

    return {
        'center': center,
        'center_info': poi_info.get(center, {'category': 'Unknown', 'cat_id': -1}),
        'center_total_weight': node_out_weight.get(center, 0),
        'neighbors': neighbors,
    }


# ============================================================
# 格式化
# ============================================================

def format_triplets_text(triplets, target_bucket):
    if not triplets:
        return ""
    lines = []
    for t in triplets:
        bname = get_bucket_name(t['bucket'])
        lines.append(
            f"(POI {t['src_poi']}[{t['src_cat']}] -> "
            f"POI {t['dst_poi']}[{t['dst_cat']}], "
            f"time: {t['dst_time']}, period: {bname})"
        )
    target_name = get_bucket_name(target_bucket)
    return (
        f"[Temporal transition patterns (target period: {target_name})]: "
        f"The following {len(triplets)} historical transitions are relevant: "
        + "; ".join(lines) + "."
    )


def format_subgraph_text(subgraphs, user_id):
    if not subgraphs:
        return ""
    parts = []
    for i, sg in enumerate(subgraphs):
        center = sg['center']
        center_cat = sg['center_info']['category']
        n_strs = []
        for n in sg['neighbors'][:10]:
            if n['out_weight'] > 0 and n['in_weight'] > 0:
                d = f"out:{n['out_weight']},in:{n['in_weight']}"
            elif n['out_weight'] > 0:
                d = f"out:{n['out_weight']}"
            else:
                d = f"in:{n['in_weight']}"
            n_strs.append(
                f"(POI {n['poi_id']}, weight: {n['total_weight']}, "
                f"{d}, category: {n['category']})"
            )
        parts.append(
            f"[Subgraph {i+1}]: Center POI {center}({center_cat}), "
            f"total outgoing weight: {sg['center_total_weight']}, "
            f"neighbors: [{'; '.join(n_strs)}]"
        )
    return (
        f"Given the user {user_id}'s POI interaction information, "
        f"learn the sequential transition relationships between POIs "
        f"based on the POI's neighbor information (weights, POI category). "
        f"The weight represents the number of times a POI transitions to "
        f"a neighboring POI in sequence, and the POI category refers to "
        f"the category of the neighboring POI. "
        + " ".join(parts)
    )


# ============================================================
# 处理单个数据集（训练集：增量构建图；测试集：查询已有图）
# ============================================================

def augment_train(
    input_path: str,
    output_path: str,
    graph: IncrementalTransitionGraph,
    max_triplets: int = 20,
    num_subgraphs: int = 2,
):
    """
    处理训练集：增量构建全局图，每个样本先查后加。
    处理完后graph包含训练集的所有转移。
    """
    print(f"[TRAIN] Loading from {input_path}...")
    dataset = load_json(input_path)
    print(f"[TRAIN] Total samples: {len(dataset)}")

    stats = _init_stats(len(dataset))
    augmented = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        question = item['question']
        answer = item['answer']

        user_id = parse_user_id_from_question(question)
        target_time_str, _ = parse_target_from_question(question)
        true_poi, true_cat = parse_answer(answer)

        if target_time_str is None or true_poi is None:
            stats['skip_parse_fail'] += 1
            augmented.append(item)
            checkins = parse_checkins_from_question(question)
            graph.add_checkins_sequence(checkins)
            continue

        target_dt = datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
        target_bucket = get_bucket(target_dt)
        stats['bucket_distribution'][get_bucket_name(target_bucket)] += 1
        current_checkins = parse_checkins_from_question(question)

        # === 先查询当前图 ===
        triplets_text, subgraph_text, triplets, subgraphs = _query_graph(
            graph, target_bucket, true_poi, user_id, max_triplets, num_subgraphs, stats
        )

        # === 再把本样本加入图 ===
        graph.add_checkins_sequence(current_checkins)
        if len(current_checkins) > 0:
            answer_checkin = {
                'time_str': target_time_str,
                'datetime': target_dt,
                'user_id': user_id,
                'poi_id': true_poi,
                'category': true_cat if true_cat else 'Unknown',
                'cat_id': -1,
            }
            graph.add_transition(current_checkins[-1], answer_checkin)

        # === 拼接 ===
        new_question = _insert_into_question(question, triplets_text, subgraph_text)
        augmented.append({'question': new_question, 'answer': answer})

        _print_progress(idx, len(dataset), graph, user_id, target_time_str,
                        target_bucket, true_poi, true_cat, triplets, subgraphs,
                        triplets_text, subgraph_text, question, new_question,
                        tag="TRAIN")

    _print_stats(stats, graph, tag="TRAIN")
    save_json(augmented, output_path)
    print(f"[TRAIN] Saved to {output_path}")
    return graph


def augment_test(
    input_path: str,
    output_path: str,
    graph: IncrementalTransitionGraph,
    max_triplets: int = 20,
    num_subgraphs: int = 2,
):
    """
    处理测试集：使用训练集构建好的完整全局图，只查询不添加。
    这是oracle实验（用了真实POI来筛选），验证可行性。
    """
    print(f"\n[TEST] Loading from {input_path}...")
    dataset = load_json(input_path)
    print(f"[TEST] Total samples: {len(dataset)}")
    print(f"[TEST] Using graph with {graph.total_edges} edges, {len(graph.poi_info)} POIs")

    stats = _init_stats(len(dataset))
    augmented = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        question = item['question']
        answer = item['answer']

        user_id = parse_user_id_from_question(question)
        target_time_str, _ = parse_target_from_question(question)
        true_poi, true_cat = parse_answer(answer)

        if target_time_str is None or true_poi is None:
            stats['skip_parse_fail'] += 1
            augmented.append(item)
            continue

        target_dt = datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
        target_bucket = get_bucket(target_dt)
        stats['bucket_distribution'][get_bucket_name(target_bucket)] += 1

        # === 只查询，不添加 ===
        triplets_text, subgraph_text, triplets, subgraphs = _query_graph(
            graph, target_bucket, true_poi, user_id, max_triplets, num_subgraphs, stats
        )

        new_question = _insert_into_question(question, triplets_text, subgraph_text)
        augmented.append({'question': new_question, 'answer': answer})

        _print_progress(idx, len(dataset), graph, user_id, target_time_str,
                        target_bucket, true_poi, true_cat, triplets, subgraphs,
                        triplets_text, subgraph_text, question, new_question,
                        tag="TEST")

    _print_stats(stats, graph, tag="TEST")
    save_json(augmented, output_path)
    print(f"[TEST] Saved to {output_path}")


# ============================================================
# 公共辅助函数
# ============================================================

def _init_stats(total):
    return {
        'total': total,
        'has_triplets': 0,
        'has_subgraphs': 0,
        'total_triplets': 0,
        'total_subgraphs': 0,
        'no_graph_edges': 0,
        'skip_parse_fail': 0,
        'bucket_distribution': defaultdict(int),
    }


def _query_graph(graph, target_bucket, true_poi, user_id,
                 max_triplets, num_subgraphs, stats):
    triplets_text = ""
    subgraph_text = ""
    triplets = []
    subgraphs = []

    if graph.total_edges > 0:
        triplets = graph.get_triplets_for_bucket(
            target_bucket, true_poi, max_triplets=max_triplets
        )
        triplets_text = format_triplets_text(triplets, target_bucket)
        if triplets:
            stats['has_triplets'] += 1
            stats['total_triplets'] += len(triplets)

        subgraphs = extract_subgraphs(
            graph.get_adj(), graph.get_poi_info(),
            true_poi, user_id, num_subgraphs=num_subgraphs
        )
        subgraph_text = format_subgraph_text(subgraphs, user_id)
        if subgraphs:
            stats['has_subgraphs'] += 1
            stats['total_subgraphs'] += len(subgraphs)
    else:
        stats['no_graph_edges'] += 1

    return triplets_text, subgraph_text, triplets, subgraphs


def _insert_into_question(question, triplets_text, subgraph_text):
    insertion_point = "Given the data,"
    if insertion_point in question:
        parts = question.split(insertion_point, 1)
        new_question = parts[0].rstrip()
        extra = ""
        if triplets_text:
            extra += " " + triplets_text
        if subgraph_text:
            extra += " " + subgraph_text
        new_question += extra + " " + insertion_point + parts[1]
    else:
        new_question = question
        if triplets_text:
            new_question = new_question.rstrip() + " " + triplets_text
        if subgraph_text:
            new_question = new_question.rstrip() + " " + subgraph_text
    return new_question


def _print_progress(idx, total, graph, user_id, target_time_str,
                    target_bucket, true_poi, true_cat, triplets, subgraphs,
                    triplets_text, subgraph_text, question, new_question,
                    tag=""):
    if (idx + 1) % 1000 == 0:
        print(f"  [{tag}] Processed {idx+1}/{total}, graph edges: {graph.total_edges}")

    if idx < 3:
        print(f"\n{'='*80}")
        print(f"[{tag}] EXAMPLE (Sample {idx})")
        print(f"{'='*80}")
        print(f"User: {user_id}, Target: {target_time_str}")
        print(f"Bucket: {get_bucket_name(target_bucket)}")
        print(f"True POI: {true_poi} ({true_cat})")
        print(f"Graph edges: {graph.total_edges}")
        print(f"Triplets found: {len(triplets)}")
        print(f"Subgraphs found: {len(subgraphs)}")
        print(f"Orig len: {len(question)}, Aug len: {len(new_question)}")
        if triplets_text:
            print(f"\nTriplets:\n  {triplets_text[:500]}")
        if subgraph_text:
            print(f"\nSubgraph:\n  {subgraph_text[:500]}")
        print(f"{'='*80}\n")


def _print_stats(stats, graph, tag=""):
    print(f"\n{'='*70}")
    print(f"[{tag}] AUGMENTATION STATISTICS")
    print(f"{'='*70}")
    print(f"Total samples:              {stats['total']}")
    print(f"Parse failures:             {stats['skip_parse_fail']}")
    print(f"No graph edges:             {stats['no_graph_edges']}")
    print(f"Graph edges:                {graph.total_edges}")
    print(f"Graph POIs:                 {len(graph.poi_info)}")
    t = max(stats['total'], 1)
    print(f"Samples with triplets:      {stats['has_triplets']} ({stats['has_triplets']/t*100:.1f}%)")
    print(f"Samples with subgraphs:     {stats['has_subgraphs']} ({stats['has_subgraphs']/t*100:.1f}%)")
    if stats['has_triplets'] > 0:
        print(f"Avg triplets/sample:        {stats['total_triplets']/stats['has_triplets']:.1f}")
    if stats['has_subgraphs'] > 0:
        print(f"Avg subgraphs/sample:       {stats['total_subgraphs']/stats['has_subgraphs']:.1f}")
    print(f"\nBucket distribution:")
    for bname, cnt in sorted(stats['bucket_distribution'].items()):
        print(f"  {bname:30s}: {cnt:6d}")
    print(f"{'='*70}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, required=True,
                        help="训练集JSON路径")
    parser.add_argument("--train_output", type=str, default=None,
                        help="增强后的训练集输出路径")
    parser.add_argument("--test_input", type=str, default=None,
                        help="测试集JSON路径（可选）")
    parser.add_argument("--test_output", type=str, default=None,
                        help="增强后的测试集输出路径")
    parser.add_argument("--max_triplets", type=int, default=20)
    parser.add_argument("--num_subgraphs", type=int, default=2)

    args = parser.parse_args()

    if args.train_output is None:
        base, ext = os.path.splitext(args.train_input)
        args.train_output = f"{base}_augmented{ext}"

    # 第1步：处理训练集，同时构建全局图
    graph = IncrementalTransitionGraph()
    graph = augment_train(
        input_path=args.train_input,
        output_path=args.train_output,
        graph=graph,
        max_triplets=args.max_triplets,
        num_subgraphs=args.num_subgraphs,
    )

    # 第2步：用训练集构建好的完整图来处理测试集
    if args.test_input is not None:
        if args.test_output is None:
            base, ext = os.path.splitext(args.test_input)
            args.test_output = f"{base}_augmented{ext}"

        augment_test(
            input_path=args.test_input,
            output_path=args.test_output,
            graph=graph,
            max_triplets=args.max_triplets,
            num_subgraphs=args.num_subgraphs,
        )