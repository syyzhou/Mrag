import json
import numpy as np
import os
import re
from collections import defaultdict, Counter

def detailed_answer_analysis():
    """详细分析答案的格式和内容"""

    data_files = {
        'train': './datasets/nyc/preprocessed/train.json',
        'test': './datasets/nyc/preprocessed/test.json'
    }

    # 收集所有答案模式
    answer_patterns = defaultdict(int)
    poi_id_patterns = []
    positive_samples = []
    negative_samples = []

    for name, filepath in data_files.items():
        if not os.path.exists(filepath):
            continue

        print(f"\n{'='*60}")
        print(f"分析 {name} 数据集答案格式")
        print(f"{'='*60}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        for entry in data:
            if isinstance(entry, dict) and 'answer' in entry:
                answer = entry['answer']

                # 分析答案长度
                answer_len = len(answer)

                # 查找POI ID
                # 尝试多种模式
                poi_ids_1 = re.findall(r'poi\s+id\s+(\d+)', answer.lower())
                poi_ids_2 = re.findall(r'poI\s+id\s+(\d+)', answer.lower())
                poi_ids_3 = re.findall(r'POI\s+id\s+(\d+)', answer.lower())
                poi_ids_4 = re.findall(r'id\s+(\d+)', answer.lower())

                all_poi_ids = []
                for id_list in [poi_ids_1, poi_ids_2, poi_ids_3, poi_ids_4]:
                    all_poi_ids.extend(id_list)

                all_poi_ids = list(set(all_poi_ids))  # 去重

                # 记录统计
                if all_poi_ids:
                    positive_samples.append({
                        'answer': answer,
                        'poi_ids': all_poi_ids,
                        'num_pois': len(all_poi_ids),
                        'answer_len': answer_len
                    })
                    poi_id_patterns.extend(all_poi_ids)
                else:
                    negative_samples.append({
                        'answer': answer,
                        'answer_len': answer_len
                    })

                # 分析答案关键词
                answer_lower = answer.lower()
                if 'next poi' in answer_lower:
                    answer_patterns['next_poi'] += 1
                if 'will visit' in answer_lower:
                    answer_patterns['will_visit'] += 1
                if 'at' in answer_lower and any(x in answer_lower for x in ['2012', '2013']):
                    answer_patterns['has_timestamp'] += 1
                if 'poi' in answer_lower and 'id' in answer_lower:
                    answer_patterns['poi_id_mentioned'] += 1
                if 'no next poi' in answer_lower or 'none' in answer_lower:
                    answer_patterns['no_next_poi'] += 1

        # 打印样本统计
        print(f"\n样本统计:")
        print(f"总记录数: {len(data)}")
        print(f"正样本数: {len(positive_samples)}")
        print(f"负样本数: {len(negative_samples)}")

        # 打印正样本示例
        print(f"\n正样本示例 (前5个):")
        for i, sample in enumerate(positive_samples[:5]):
            print(f"\n示例 {i+1}:")
            print(f"  答案: {sample['answer'][:100]}...")
            print(f"  POI IDs: {sample['poi_ids']}")
            print(f"  POI数量: {sample['num_pois']}")
            print(f"  答案长度: {sample['answer_len']}")

        # 打印负样本示例
        print(f"\n负样本示例 (前5个):")
        for i, sample in enumerate(negative_samples[:5]):
            print(f"\n示例 {i+1}:")
            print(f"  答案: {sample['answer'][:100]}...")
            print(f"  答案长度: {sample['answer_len']}")

        # POI ID分布
        if poi_id_patterns:
            print(f"\nPOI ID分布统计:")
            poi_counter = Counter(poi_id_patterns)
            print(f"唯一POI ID数: {len(poi_counter)}")
            print(f"最常出现的10个POI ID:")
            for poi_id, count in poi_counter.most_common(10):
                print(f"  POI {poi_id}: {count} 次")

        # 答案模式统计
        print(f"\n答案关键词统计:")
        for pattern, count in answer_patterns.items():
            print(f"  {pattern}: {count} 次")

        # 计算正负样本比例
        total_samples = len(positive_samples) + len(negative_samples)
        if total_samples > 0:
            pos_ratio = len(positive_samples) / total_samples * 100
            neg_ratio = len(negative_samples) / total_samples * 100
            print(f"\n样本比例:")
            print(f"  正样本: {pos_ratio:.1f}% ({len(positive_samples)})")
            print(f"  负样本: {neg_ratio:.1f}% ({len(negative_samples)})")

    # 分析大文件
    print(f"\n{'='*60}")
    print("分析大型QA数据集")
    print(f"{'='*60}")

    large_files = [
        ('train_large', './datasets/nyc/preprocessed/train_qa_pairs_kqt.json'),
        ('test_large', './datasets/nyc/preprocessed/test_qa_pairs_kqt.json')
    ]

    for name, filepath in large_files:
        if not os.path.exists(filepath):
            continue

        print(f"\n分析 {name} 数据集:")
        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"记录数: {len(data)}")

        # 分析前几个样本
        positive_count = 0
        negative_count = 0

        for i, entry in enumerate(data[:100]):  # 只分析前100个
            if isinstance(entry, dict) and 'answer' in entry:
                answer = entry['answer']
                poi_ids = re.findall(r'poi\s+id\s+(\d+)', answer.lower())

                if poi_ids:
                    positive_count += 1
                else:
                    negative_count += 1

        print(f"前100个样本中:")
        print(f"  正样本: {positive_count}")
        print(f"  负样本: {negative_count}")


def create_visualization_data():
    """创建适合用图展示的数据"""

    print(f"\n{'='*60}")
    print("创建可视化数据")
    print(f"{'='*60}")

    # 分析数据集大小
    data_info = {
        'datasets': {}
    }

    files_to_analyze = {
        'train_small': './datasets/nyc/preprocessed/train.json',
        'test_small': './datasets/nyc/preprocessed/test.json',
        'train_large': './datasets/nyc/preprocessed/train_qa_pairs_kqt.json',
        'test_large': './datasets/nyc/preprocessed/test_qa_pairs_kqt.json'
    }

    for name, filepath in files_to_analyze.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)

            # 计算基本统计
            if isinstance(data, list):
                # 分析正负样本
                positive_count = 0
                negative_count = 0
                user_pois = defaultdict(set)

                for entry in data[:1000]:  # 限制分析数量
                    if isinstance(entry, dict):
                        if 'answer' in entry:
                            answer = entry['answer']
                            poi_ids = re.findall(r'poi\s+id\s+(\d+)', answer.lower())
                            if poi_ids:
                                positive_count += 1
                                user_id = str(entry.get('user_id', 'unknown'))
                                for pid in poi_ids:
                                    user_pois[user_id].add(pid)
                            else:
                                negative_count += 1

                data_info['datasets'][name] = {
                    'total_records': len(data),
                    'positive_samples': positive_count,
                    'negative_samples': negative_count,
                    'unique_users': len(user_pois),
                    'avg_pois_per_user': len([u for u in user_pois.values()]) / len(user_pois) if user_pois else 0
                }

                print(f"{name}:")
                print(f"  总记录: {len(data)}")
                print(f"  正样本: {positive_count}")
                print(f"  负样本: {negative_count}")
                print(f"  独立用户: {len(user_pois)}")

    # 保存结果
    with open('visualization_data.json', 'w') as f:
        json.dump(data_info, f, indent=2)

    print(f"\n可视化数据已保存为: visualization_data.json")

    # 创建ASCII图表
    print(f"\nASCII图表展示:")
    print("-" * 60)

    # 数据集大小对比
    print("\n数据集大小对比:")
    max_size = max([d['total_records'] for d in data_info['datasets'].values()])

    for name, info in data_info['datasets'].items():
        bar_length = int(info['total_records'] / max_size * 50)
        bar = '#' * bar_length
        print(f"{name:15} {bar} {info['total_records']:,}")

    # 正负样本比例
    print("\n正负样本比例:")
    for name, info in data_info['datasets'].items():
        total = info['positive_samples'] + info['negative_samples']
        if total > 0:
            pos_bar = '#' * int(info['positive_samples'] / total * 30)
            neg_bar = '#' * int(info['negative_samples'] / total * 30)
            print(f"{name:15} 正样本 {pos_bar} {info['positive_samples']}")
            print(f"               负样本 {neg_bar} {info['negative_samples']}")


if __name__ == "__main__":
    detailed_answer_analysis()
    create_visualization_data()