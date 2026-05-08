import json
import re
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
import random

def parse_user_and_time_from_question(question_text):
    """从问题文本中解析用户ID和目标时间"""
    # 解析用户ID - 适配多种格式: "user 123", "user_123", "Which POI id will user 123 visit"
    user_match = re.search(r'user[_\s]?(\d+)', question_text, re.IGNORECASE)
    user_id = int(user_match.group(1)) if user_match else None
    
    # 解析目标时间 - 取最后一个 "At YYYY-MM-DD HH:MM:SS"
    time_pattern = r'At (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})'
    matches = re.findall(time_pattern, question_text)
    target_time = None
    if matches:
        date_str, time_str = matches[-1]
        target_time = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
    
    return user_id, target_time


def parse_label_poi_from_answer(answer_text):
    """从答案中解析正确的POI ID"""
    match = re.search(r'POI id (\d+)', answer_text)
    return int(match.group(1)) if match else None


def build_user_visit_history(csv_paths):
    """
    从CSV文件构建每个用户的访问历史
    
    Args:
        csv_paths: 单个路径或路径列表
    
    Returns:
        {user_id: [(poi_id, datetime_obj), ...]} 按时间排序
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    all_visits = defaultdict(list)
    
    for csv_path in csv_paths:
        print(f"  读取: {csv_path}")
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            user_id = int(row['UserId'])
            poi_id = int(row['PoiId'])
            
            # 解析时间 - 适配不同格式
            if 'UTCTimeOffset' in df.columns:
                # 格式: "2012-04-03 18:00:09+00:00" 或 "2012-04-03 18:00:09"
                time_str = str(row['UTCTimeOffset'])
                # 去掉时区部分
                time_str = re.sub(r'[+-]\d{2}:\d{2}$', '', time_str)
                try:
                    timestamp = datetime.strptime(time_str.strip(), '%Y-%m-%d %H:%M:%S')
                except:
                    timestamp = datetime.strptime(time_str.strip()[:19], '%Y-%m-%d %H:%M:%S')
            else:
                # 使用epoch时间
                timestamp = datetime.fromtimestamp(row['UTCTimeOffsetEpoch'])
            
            all_visits[user_id].append((poi_id, timestamp))
    
    # 按时间排序
    for user_id in all_visits:
        all_visits[user_id].sort(key=lambda x: x[1])
    
    print(f"  共 {len(all_visits)} 个用户")
    total_visits = sum(len(v) for v in all_visits.values())
    print(f"  共 {total_visits} 条访问记录")
    
    return all_visits


def get_candidate_pois_for_sample(user_id, target_time, user_visits, label_poi, max_candidates=20):
    """
    为单个样本获取候选POI列表
    
    规则:
    1. 统计用户在target_time之前的所有访问POI及频次
    2. 按频次降序排列
    3. 如果超过max_candidates，按频次取top，但必须包含label_poi
    4. 最终列表必须包含label_poi
    
    Returns:
        候选POI列表
    """
    if user_id not in user_visits:
        # 用户没有历史记录
        return [label_poi] if label_poi is not None else []
    
    # 统计目标时间之前的所有访问
    poi_counter = Counter()
    for poi_id, visit_time in user_visits[user_id]:
        if visit_time < target_time:
            poi_counter[poi_id] += 1
    
    if not poi_counter:
        # 该时间之前没有访问记录
        return [label_poi] if label_poi is not None else []
    
    # 按频次降序排列
    sorted_pois = [poi for poi, count in poi_counter.most_common()]
    
    # 处理候选列表
    if len(sorted_pois) <= max_candidates:
        # 不需要截断
        candidates = sorted_pois.copy()
        # 确保包含label_poi
        if label_poi is not None and label_poi not in candidates:
            candidates.append(label_poi)
    else:
        # 需要截断到max_candidates
        if label_poi is not None and label_poi in sorted_pois[:max_candidates]:
            # label已在top中
            candidates = sorted_pois[:max_candidates]
        elif label_poi is not None:
            # label不在top中，需要加入
            candidates = sorted_pois[:max_candidates - 1]
            candidates.append(label_poi)
        else:
            candidates = sorted_pois[:max_candidates]
    
    return candidates


def process_txt_file(
    original_txt_path,
    output_txt_path,
    user_visits,
    max_candidates=20,
    shuffle_candidates=True,
    seed=42
):
    """
    处理TXT格式的QA文件
    """
    random.seed(seed)
    
    print(f"\n处理TXT文件: {original_txt_path}")
    
    with open(original_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    stats = {
        'total': 0,
        'success': 0,
        'no_user': 0,
        'no_time': 0,
        'no_label': 0,
        'no_history': 0,
        'avg_candidates': []
    }
    
    for idx, line in enumerate(lines):
        line = line.strip()
        stats['total'] += 1
        
        if "<answer>:" not in line:
            new_lines.append(line)
            continue
        
        # 分离question和answer
        parts = line.split("<answer>:")
        question_part = parts[0]
        answer_part = parts[1] if len(parts) > 1 else ""
        
        # 解析用户ID和时间
        user_id, target_time = parse_user_and_time_from_question(question_part)
        
        if user_id is None:
            stats['no_user'] += 1
            new_lines.append(line)
            continue
            
        if target_time is None:
            stats['no_time'] += 1
            new_lines.append(line)
            continue
        
        # 解析label POI
        label_poi = parse_label_poi_from_answer(answer_part)
        
        if label_poi is None:
            stats['no_label'] += 1
            new_lines.append(line)
            continue
        
        # 获取候选POI
        candidates = get_candidate_pois_for_sample(
            user_id, target_time, user_visits, label_poi, max_candidates
        )
        
        if len(candidates) <= 1:
            stats['no_history'] += 1
            # 仍然添加，即使只有label一个候选
        
        stats['success'] += 1
        stats['avg_candidates'].append(len(candidates))
        
        # 打乱顺序（可选）
        if shuffle_candidates:
            random.shuffle(candidates)
        
        # 构造候选字符串
        candidate_str = (
                f" Use the following candidate POIs as supplementary references "
                f"to refine your prediction. Candidate POIs:{candidates}."
        )
        
        # 插入到 <answer>: 前
        new_line = f"{question_part}{candidate_str} <answer>:{answer_part}"
        new_lines.append(new_line)
    
    # 保存
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines) + '\n')
    
    # 打印统计
    print(f"\n  统计信息:")
    print(f"    总样本数: {stats['total']}")
    print(f"    成功处理: {stats['success']}")
    print(f"    无用户ID: {stats['no_user']}")
    print(f"    无时间信息: {stats['no_time']}")
    print(f"    无label: {stats['no_label']}")
    print(f"    无历史记录: {stats['no_history']}")
    if stats['avg_candidates']:
        avg = sum(stats['avg_candidates']) / len(stats['avg_candidates'])
        print(f"    平均候选数: {avg:.2f}")
    print(f"\n  保存到: {output_txt_path}")
    
    return stats


def process_json_file(
    original_json_path,
    output_json_path,
    user_visits,
    max_candidates=20,
    shuffle_candidates=True,
    seed=42
):
    """
    处理JSON格式的QA文件
    """
    random.seed(seed)
    
    print(f"\n处理JSON文件: {original_json_path}")
    
    with open(original_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'total': 0,
        'success': 0,
        'no_user': 0,
        'no_time': 0,
        'no_label': 0,
        'no_history': 0,
        'avg_candidates': []
    }
    
    new_data = []
    
    for item in data:
        stats['total'] += 1
        
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # 解析用户ID和时间
        user_id, target_time = parse_user_and_time_from_question(question)
        
        if user_id is None:
            stats['no_user'] += 1
            new_data.append(item)
            continue
            
        if target_time is None:
            stats['no_time'] += 1
            new_data.append(item)
            continue
        
        # 解析label POI
        label_poi = parse_label_poi_from_answer(answer)
        
        if label_poi is None:
            stats['no_label'] += 1
            new_data.append(item)
            continue
        
        # 获取候选POI
        candidates = get_candidate_pois_for_sample(
            user_id, target_time, user_visits, label_poi, max_candidates
        )
        
        if len(candidates) <= 1:
            stats['no_history'] += 1
        
        stats['success'] += 1
        stats['avg_candidates'].append(len(candidates))
        
        # 打乱顺序（可选）
        if shuffle_candidates:
            random.shuffle(candidates)
        
        # 构造新的question
        candidate_str = (
            f" Use the following candidate POIs as supplementary references "
            f"to refine your prediction. Candidate POIs:{candidates}."
        )
        new_question = question + candidate_str
        
        # 创建新item
        new_item = item.copy()
        new_item['question'] = new_question
        new_item['candidates'] = candidates  # 额外保存候选列表
        new_data.append(new_item)
    
    # 保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print(f"\n  统计信息:")
    print(f"    总样本数: {stats['total']}")
    print(f"    成功处理: {stats['success']}")
    print(f"    无用户ID: {stats['no_user']}")
    print(f"    无时间信息: {stats['no_time']}")
    print(f"    无label: {stats['no_label']}")
    print(f"    无历史记录: {stats['no_history']}")
    if stats['avg_candidates']:
        avg = sum(stats['avg_candidates']) / len(stats['avg_candidates'])
        print(f"    平均候选数: {avg:.2f}")
    print(f"\n  保存到: {output_json_path}")
    
    return stats


def main():
    # ============ 配置路径 ============
    # 原始CSV数据
    train_csv_path = "../datasets/nyc/preprocessed/train_sample.csv"
    test_csv_path = "../datasets/nyc/preprocessed/test_sample.csv"
    
    # 问答格式文件
    train_qa_json_path = "../datasets/nyc/preprocessed/train_qa_pairs_kqt.json"
    test_qa_txt_path = "../datasets/nyc/preprocessed/test_qa_pairs_kqt.txt"
    
    # 输出文件
    train_output_json_path = "../datasets/nyc/preprocessed/train_qa_pairs_kqt_candidates.json"
    test_output_txt_path = "../datasets/nyc/preprocessed/test_qa_pairs_kqt_candidates.txt"
    
    # 参数
    max_candidates = 20
    shuffle_candidates = True
    seed = 42
    
    # ============ 步骤1: 构建用户访问历史 ============
    print("=" * 60)
    print("步骤1: 构建用户访问历史")
    print("=" * 60)
    
    # 对于训练集：只用训练集的数据
    print("\n[训练集历史] 只使用训练集数据")
    train_user_visits = build_user_visit_history(train_csv_path)
    
    # 对于测试集：使用训练集 + 测试集的数据（测试时可以看到训练集的历史）
    print("\n[测试集历史] 使用训练集 + 测试集数据")
    test_user_visits = build_user_visit_history([train_csv_path, test_csv_path])
    
    # ============ 步骤2: 处理训练集 ============
    print("\n" + "=" * 60)
    print("步骤2: 处理训练集")
    print("=" * 60)
    
    process_json_file(
        train_qa_json_path,
        train_output_json_path,
        train_user_visits,  # 训练集只用训练集历史
        max_candidates=max_candidates,
        shuffle_candidates=shuffle_candidates,
        seed=seed
    )
    
    # ============ 步骤3: 处理测试集 ============
    print("\n" + "=" * 60)
    print("步骤3: 处理测试集")
    print("=" * 60)
    
    process_txt_file(
        test_qa_txt_path,
        test_output_txt_path,
        test_user_visits,  # 测试集用完整历史
        max_candidates=max_candidates,
        shuffle_candidates=shuffle_candidates,
        seed=seed
    )
    
    print("\n" + "=" * 60)
    print("✓ 全部处理完成!")
    print("=" * 60)


def verify_sample(txt_path, num_samples=3):
    """验证几个样本的结果"""
    print(f"\n验证样本 ({txt_path}):")
    print("-" * 60)
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines[:num_samples]):
        print(f"\n样本 {i+1}:")
        
        # 提取候选POI
        candidate_match = re.search(r'Candidate POIs:\[(.*?)\]', line)
        if candidate_match:
            candidates = candidate_match.group(1)
            print(f"  候选POI: [{candidates}]")
        
        # 提取label
        label_match = re.search(r'visit POI id (\d+)', line)
        if label_match:
            label = label_match.group(1)
            print(f"  正确答案: POI id {label}")
            
            # 检查label是否在候选中
            if candidate_match and label in candidates:
                print(f"  ✓ Label在候选中")
            else:
                print(f"  ✗ Label不在候选中!")


if __name__ == "__main__":
    main()
    
    # 验证结果
    verify_sample("../datasets/nyc/preprocessed/test_qa_pairs_kqt_candidates.txt", num_samples=5)