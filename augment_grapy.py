# augment_test_with_graph.py
"""
只处理测试集：
1. 先从训练集构建完整全局图（不输出增强训练集）
2. 用完整图处理测试集（oracle：用真实POI筛选）
3. 支持txt格式（每行一个json）
"""

import json
import re
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional


# ============================================================
# IO
# ============================================================
def load_data(path):
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # txt: 每行是 <question>:...<answer>:... 格式
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 按 <answer>: 分割
                if '<answer>:' in line:
                    parts = line.split('<answer>:', 1)
                    question = parts[0].strip()
                    answer = '<answer>:' + parts[1].strip()
                    data.append({
                        'question': question,
                        'answer': answer,
                    })
                else:
                    # 无法解析的行跳过
                    print(f"  WARNING: cannot parse line: {line[:100]}...")
        return data


def save_data(data, path):
    if path.endswith('.json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        # txt: 还原为 question + answer 一行
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                question = item['question']
                answer = item['answer']
                # answer已经带<answer>:前缀就直接拼，否则加上
                if not answer.startswith('<answer>:'):
                    line = question + '<answer>:' + answer
                else:
                    line = question + answer
                f.write(line + '\n')
# ============================================================
# 时间分桶 (8桶)
# ============================================================

PERIOD_NAMES = ['morning(06-11)', 'noon(11-14)', 'afternoon(14-18)', 'evening(18-06)']


def get_period(hour):
    if 6 <= hour < 11: return 0
    elif 11 <= hour < 14: return 1
    elif 14 <= hour < 18: return 2
    else: return 3


def get_bucket(dt):
    period = get_period(dt.hour)
    return period + 4 if dt.weekday() >= 5 else period


def get_bucket_name(bucket_id):
    day_type = "weekend" if bucket_id >= 4 else "weekday"
    return f"{day_type}_{PERIOD_NAMES[bucket_id % 4]}"


def get_adjacent_buckets(bucket_id):
    day_offset = 4 if bucket_id >= 4 else 0
    period = bucket_id % 4
    adj = []
    if period > 0: adj.append(day_offset + period - 1)
    if period < 3: adj.append(day_offset + period + 1)
    adj.append(((day_offset + 4) % 8) + period)
    return adj


# ============================================================
# 文本解析
# ============================================================

def parse_checkins_from_question(question):
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
            'time_str': time_str, 'datetime': dt,
            'user_id': int(m.group(2)), 'poi_id': int(m.group(3)),
            'category': m.group(4), 'cat_id': int(m.group(5)),
        })
    checkins.sort(key=lambda x: x['datetime'])
    return checkins


def parse_target_from_question(question):
    m = re.search(
        r'At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), Which POI id will user (\d+) visit',
        question)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def parse_answer(answer):
    m = re.search(r'will visit POI id (\d+)\.(.+?)\.?$', answer.strip())
    if m:
        return int(m.group(1)), m.group(2).strip().rstrip('.')
    m2 = re.search(r'POI id (\d+)', answer)
    return (int(m2.group(1)), "") if m2 else (None, None)


def parse_user_id_from_question(question):
    m = re.search(r'user (\d+)', question)
    return int(m.group(1)) if m else None


# ============================================================
# 全局转移图
# ============================================================

class IncrementalTransitionGraph:
    def __init__(self):
        self.adj = defaultdict(lambda: defaultdict(lambda: {
            'weight': 0, 'category': '', 'cat_id': -1}))
        self.poi_info = {}
        self.bucket_transitions = defaultdict(list)
        self.total_edges = 0

    def add_transition(self, src_checkin, dst_checkin):
        src, dst = src_checkin['poi_id'], dst_checkin['poi_id']
        self.poi_info[src] = {
            'category': src_checkin['category'],
            'cat_id': src_checkin.get('cat_id', -1)}
        self.poi_info[dst] = {
            'category': dst_checkin['category'],
            'cat_id': dst_checkin.get('cat_id', -1)}
        self.adj[src][dst]['weight'] += 1
        self.adj[src][dst]['category'] = dst_checkin['category']
        self.adj[src][dst]['cat_id'] = dst_checkin.get('cat_id', -1)
        bucket = get_bucket(dst_checkin['datetime'])
        self.bucket_transitions[bucket].append({
            'src_poi': src, 'src_cat': src_checkin['category'],
            'src_cat_id': src_checkin.get('cat_id', -1),
            'dst_poi': dst, 'dst_cat': dst_checkin['category'],
            'dst_cat_id': dst_checkin.get('cat_id', -1),
            'dst_time': dst_checkin['time_str'],
            'dst_dt': dst_checkin['datetime'], 'bucket': bucket,
        })
        self.total_edges += 1

    def add_checkins_sequence(self, checkins):
        for i in range(len(checkins) - 1):
            self.add_transition(checkins[i], checkins[i + 1])

    def get_adj(self):
        return dict(self.adj)

    def get_poi_info(self):
        return dict(self.poi_info)

    def get_triplets_for_bucket(self, bucket_id, true_poi, max_triplets=20):
        def contains_true(t):
            return t['src_poi'] == true_poi or t['dst_poi'] == true_poi

        matching = [t for t in self.bucket_transitions[bucket_id]
                    if contains_true(t)]

        if len(matching) < max_triplets:
            for adj_b in get_adjacent_buckets(bucket_id):
                for t in self.bucket_transitions[adj_b]:
                    if contains_true(t) and t not in matching:
                        matching.append(t)
                        if len(matching) >= max_triplets: break
                if len(matching) >= max_triplets: break

        if len(matching) < max_triplets:
            for b, trans_list in self.bucket_transitions.items():
                for t in trans_list:
                    if contains_true(t) and t not in matching:
                        matching.append(t)
                        if len(matching) >= max_triplets: break
                if len(matching) >= max_triplets: break

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
    subgraphs, used = [], set()

    for center in sorted_nodes:
        if len(subgraphs) >= num_subgraphs: break
        if center in used: continue
        out_n = set(adj.get(center, {}).keys())
        in_n = set(in_adj.get(center, {}).keys())
        if center != true_poi and true_poi not in (out_n | in_n):
            continue
        subgraphs.append(_build_sg(center, adj, in_adj, poi_info, node_out_weight))
        used.add(center)

    if len(subgraphs) < num_subgraphs:
        for center in sorted_nodes:
            if len(subgraphs) >= num_subgraphs: break
            if center in used: continue
            one_hop = set(adj.get(center, {}).keys()) | set(in_adj.get(center, {}).keys())
            found = any(true_poi in (set(adj.get(n1, {}).keys()) |
                        set(in_adj.get(n1, {}).keys())) for n1 in one_hop)
            if not found: continue
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
            'out_weight': wt['out_weight'], 'in_weight': wt['in_weight'],
            'category': info['category'], 'cat_id': info['cat_id'],
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
    if not triplets: return ""
    lines = []
    for t in triplets:
        lines.append(
            f"(POI {t['src_poi']}[{t['src_cat']}] -> "
            f"POI {t['dst_poi']}[{t['dst_cat']}], "
            f"time: {t['dst_time']}, period: {get_bucket_name(t['bucket'])})")
    return (
        f"[Temporal transition patterns (target period: {get_bucket_name(target_bucket)})]: "
        f"The following {len(triplets)} historical transitions are relevant: "
        + "; ".join(lines) + ".")


def format_subgraph_text(subgraphs, user_id):
    if not subgraphs: return ""
    parts = []
    for i, sg in enumerate(subgraphs):
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
                f"{d}, category: {n['category']})")
        parts.append(
            f"[Subgraph {i+1}]: Center POI {sg['center']}"
            f"({sg['center_info']['category']}), "
            f"total outgoing weight: {sg['center_total_weight']}, "
            f"neighbors: [{'; '.join(n_strs)}]")
    return (
        f"Given the user {user_id}'s POI interaction information, "
        f"learn the sequential transition relationships between POIs "
        f"based on the POI's neighbor information (weights, POI category). "
        f"The weight represents the number of times a POI transitions to "
        f"a neighboring POI in sequence, and the POI category refers to "
        f"the category of the neighboring POI. " + " ".join(parts))


def insert_into_question(question, triplets_text, subgraph_text):
    insertion_point = "Given the data,"
    if insertion_point in question:
        parts = question.split(insertion_point, 1)
        new_q = parts[0].rstrip()
        extra = ""
        if triplets_text: extra += " " + triplets_text
        if subgraph_text: extra += " " + subgraph_text
        return new_q + extra + " " + insertion_point + parts[1]
    else:
        new_q = question
        if triplets_text: new_q = new_q.rstrip() + " " + triplets_text
        if subgraph_text: new_q = new_q.rstrip() + " " + subgraph_text
        return new_q


# ============================================================
# 第1步：从训练集构建完整全局图
# ============================================================

def build_graph_from_train(train_path):
    """从训练集构建完整全局图（增量方式，和augment_dataset一致）"""
    print(f"[GRAPH] Loading training data from {train_path}...")
    dataset = load_data(train_path)
    print(f"[GRAPH] Training samples: {len(dataset)}")

    graph = IncrementalTransitionGraph()

    for idx, item in enumerate(dataset):
        question = item['question']
        answer = item['answer']

        user_id = parse_user_id_from_question(question)
        target_time_str, _ = parse_target_from_question(question)
        true_poi, true_cat = parse_answer(answer)
        current_checkins = parse_checkins_from_question(question)

        # 加入question中的转移
        graph.add_checkins_sequence(current_checkins)

        # 加入answer中的转移
        if (true_poi is not None and target_time_str is not None
                and len(current_checkins) > 0):
            target_dt = datetime.strptime(target_time_str, '%Y-%m-%d %H:%M:%S')
            answer_checkin = {
                'time_str': target_time_str, 'datetime': target_dt,
                'user_id': user_id, 'poi_id': true_poi,
                'category': true_cat if true_cat else 'Unknown',
                'cat_id': -1,
            }
            graph.add_transition(current_checkins[-1], answer_checkin)

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx+1}/{len(dataset)}, edges: {graph.total_edges}")

    print(f"[GRAPH] Complete. Edges: {graph.total_edges}, POIs: {len(graph.poi_info)}")

    # 打印分桶统计
    for b in range(8):
        n = len(graph.bucket_transitions[b])
        print(f"  Bucket {b} ({get_bucket_name(b)}): {n} transitions")

    return graph


# ============================================================
# 第2步：用完整图处理测试集
# ============================================================

def augment_test(
    test_path, output_path, graph,
    max_triplets=20, num_subgraphs=2,
):
    print(f"\n[TEST] Loading test data from {test_path}...")
    dataset = load_data(test_path)
    print(f"[TEST] Test samples: {len(dataset)}")
    print(f"[TEST] Using graph: {graph.total_edges} edges, {len(graph.poi_info)} POIs")

    stats = {
        'total': len(dataset), 'has_triplets': 0, 'has_subgraphs': 0,
        'total_triplets': 0, 'total_subgraphs': 0,
        'skip_parse_fail': 0, 'bucket_distribution': defaultdict(int),
    }

    augmented = []

    for idx, item in enumerate(dataset):
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
        triplets = graph.get_triplets_for_bucket(
            target_bucket, true_poi, max_triplets=max_triplets)
        triplets_text = format_triplets_text(triplets, target_bucket)
        if triplets:
            stats['has_triplets'] += 1
            stats['total_triplets'] += len(triplets)

        subgraphs = extract_subgraphs(
            graph.get_adj(), graph.get_poi_info(),
            true_poi, user_id, num_subgraphs=num_subgraphs)
        subgraph_text = format_subgraph_text(subgraphs, user_id)
        if subgraphs:
            stats['has_subgraphs'] += 1
            stats['total_subgraphs'] += len(subgraphs)

        new_question = insert_into_question(question, triplets_text, subgraph_text)

        result = {'question': new_question, 'answer': answer}
        if 'candidates' in item:
            result['candidates'] = item['candidates']
        augmented.append(result)

        if (idx + 1) % 1000 == 0:
            print(f"  [TEST] Processed {idx+1}/{len(dataset)}")

        if idx < 3:
            print(f"\n{'='*80}")
            print(f"[TEST] Sample {idx}")
            print(f"{'='*80}")
            print(f"User: {user_id}, Target: {target_time_str}")
            print(f"Bucket: {get_bucket_name(target_bucket)}")
            print(f"True POI: {true_poi} ({true_cat})")
            print(f"Triplets: {len(triplets)}, Subgraphs: {len(subgraphs)}")
            print(f"Orig len: {len(question)}, Aug len: {len(new_question)}")
            if triplets_text:
                print(f"\nTriplets:\n  {triplets_text[:400]}...")
            if subgraph_text:
                print(f"\nSubgraph:\n  {subgraph_text[:400]}...")
            print(f"{'='*80}\n")

    # 统计
    t = max(stats['total'], 1)
    print(f"\n{'='*70}")
    print("[TEST] AUGMENTATION STATISTICS")
    print(f"{'='*70}")
    print(f"Total samples:          {stats['total']}")
    print(f"Parse failures:         {stats['skip_parse_fail']}")
    print(f"With triplets:          {stats['has_triplets']} ({stats['has_triplets']/t*100:.1f}%)")
    print(f"With subgraphs:         {stats['has_subgraphs']} ({stats['has_subgraphs']/t*100:.1f}%)")
    if stats['has_triplets'] > 0:
        print(f"Avg triplets/sample:    {stats['total_triplets']/stats['has_triplets']:.1f}")
    if stats['has_subgraphs'] > 0:
        print(f"Avg subgraphs/sample:   {stats['total_subgraphs']/stats['has_subgraphs']:.1f}")
    print(f"\nBucket distribution:")
    for bname, cnt in sorted(stats['bucket_distribution'].items()):
        print(f"  {bname:30s}: {cnt:6d}")
    print(f"{'='*70}")

    save_data(augmented, output_path)
    print(f"\n[TEST] Saved to {output_path}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="用训练集的全局图增强测试集")
    parser.add_argument("--train_input", type=str, required=True,
                        help="训练集路径 (用于构建全局图)")
    parser.add_argument("--test_input", type=str, required=True,
                        help="测试集路径 (.txt)")
    parser.add_argument("--test_output", type=str, default=None,
                        help="增强后的测试集输出路径")
    parser.add_argument("--max_triplets", type=int, default=20)
    parser.add_argument("--num_subgraphs", type=int, default=2)

    args = parser.parse_args()

    if args.test_output is None:
        base, ext = os.path.splitext(args.test_input)
        args.test_output = f"{base}_augmented{ext}"

    # 第1步：从训练集构建全局图
    graph = build_graph_from_train(args.train_input)

    # 第2步：用全局图增强测试集
    augment_test(
        test_path=args.test_input,
        output_path=args.test_output,
        graph=graph,
        max_triplets=args.max_triplets,
        num_subgraphs=args.num_subgraphs,
    )