import json
import numpy as np
import os
from collections import defaultdict, Counter

def analyze_qa_dataset():
    """分析QA数据集中的正负样本分布"""

    # 分析一些大的QA数据集文件
    qa_files = [
        './datasets/nyc/preprocessed/train_qa_pairs_kqt.json',
        './datasets/nyc/preprocessed/test_qa_pairs_kqt.json'
    ]

    for filepath in qa_files:
        if not os.path.exists(filepath):
            print(f"警告: {filepath} 不存在")
            continue

        print(f"\n{'='*60}")
        print(f"分析文件: {os.path.basename(filepath)}")
        print(f"{'='*60}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"数据类型: {type(data)}")
        print(f"数据大小: {len(data)} 条记录")

        if isinstance(data, list):
            # 分析列表格式的数据
            positive_counts = []
            negative_counts = []
            user_stats = defaultdict(lambda: {'pos': 0, 'neg': 0})

            print("\n分析前10条记录结构:")
            for i, entry in enumerate(data[:10]):
                print(f"\n记录 {i+1}:")
                print(f"  类型: {type(entry)}")
                print(f"  键: {list(entry.keys()) if isinstance(entry, dict) else 'N/A'}")

                if isinstance(entry, dict):
                    # 检查答案
                    answer = entry.get('answer', '')
                    has_positive = 'next poi id' in answer.lower()

                    if has_positive:
                        # 尝试提取POI ID
                        import re
                        poi_ids = re.findall(r'poi\s+id\s+(\d+)', answer.lower())
                        if poi_ids:
                            positive_counts.append(len(poi_ids))
                            # 假设这是正样本
                            user_id = entry.get('user_id', str(hash(str(entry))))
                            user_stats[user_id]['pos'] += 1
                    else:
                        user_id = entry.get('user_id', str(hash(str(entry))))
                        user_stats[user_id]['neg'] += 1

            print(f"\n样本统计:")
            print(f"正样本数量: {len(positive_counts)}")
            print(f"负样本数量: {len([u for u in user_stats.values() if u['neg'] > 0])}")
            print(f"用户总数: {len(user_stats)}")

            if positive_counts:
                print(f"\n正样本分布:")
                print(f"平均每用户正样本数: {np.mean(positive_counts):.2f}")
                print(f"最大正样本数: {max(positive_counts)}")
                print(f"最小正样本数: {min(positive_counts)}")

                # 百分位数
                percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
                print("\n百分位数:")
                for p in percentiles:
                    value = np.percentile(positive_counts, p)
                    print(f"  {p}%: {value:.0f}")

                # 直方图
                print("\n直方图分布:")
                hist_data = positive_counts
                max_val = max(hist_data)
                if max_val <= 20:
                    bins = list(range(0, max_val + 2, 1))
                else:
                    step = max(1, max_val // 20)
                    bins = list(range(0, max_val + step, step))

                hist, _ = np.histogram(hist_data, bins=bins)
                max_count = max(hist) if hist else 1

                for i in range(len(hist)):
                    start = bins[i]
                    end = bins[i+1] - 1 if i < len(hist)-1 else bins[i+1]
                    bar_length = hist[i] * 40 // max_count
                    bar = '#' * bar_length
                    print(f"  {start:2d}-{end:2d}: {hist[i]:3d} 用户 {bar}")

            # TopK建议
            print("\nTopK 覆盖分析:")
            all_positive = list(positive_counts) if positive_counts else [1] * len(user_stats)
            topk_values = [5, 10, 20, 50, 100, 200]

            for k in topk_values:
                coverage = sum(1 for count in all_positive if count <= k) / len(all_positive) * 100
                print(f"  TopK={k}: {coverage:.1f}% 用户正样本数 ≤ {k}")

        # 保存详细统计
        if positive_counts:
            stats = {
                'file': os.path.basename(filepath),
                'total_records': len(data),
                'num_users': len(user_stats),
                'num_positive_samples': len(positive_counts),
                'mean_positives_per_user': float(np.mean(positive_counts)),
                'median_positives_per_user': float(np.median(positive_counts)),
                'max_positives_per_user': int(max(positive_counts)),
                'min_positives_per_user': int(min(positive_counts)),
                'percentiles': {f"{p}%": float(np.percentile(positive_counts, p)) for p in percentiles},
                'positive_counts': positive_counts
            }

            output_file = f"{os.path.basename(filepath).replace('.json', '_analysis.json')}"
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n详细统计已保存为: {output_file}")


def analyze_smaller_dataset():
    """分析之前看到的3个条目的数据集"""

    data_files = {
        'train': './datasets/nyc/preprocessed/train.json',
        'test': './datasets/nyc/preprocessed/test.json'
    }

    print(f"\n{'='*60}")
    print("分析小型数据集")
    print(f"{'='*60}")

    for name, filepath in data_files.items():
        if not os.path.exists(filepath):
            continue

        print(f"\n{name} 数据集分析:")
        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"记录数: {len(data)}")

        # 分析每个记录的正样本数量
        positive_counts_per_record = []

        for i, entry in enumerate(data):
            if isinstance(entry, dict) and 'answer' in entry:
                answer = entry['answer']
                # 查找答案中的POI ID
                import re
                poi_ids = re.findall(r'poi\s+id\s+(\d+)', answer.lower())
                positive_counts_per_record.append(len(poi_ids))

                print(f"\n记录 {i+1}:")
                print(f"  用户ID: {entry.get('user_id', 'unknown')}")
                print(f"  答案: {answer[:100]}...")
                print(f"  正样本数: {len(poi_ids)}")

        print(f"\n记录级正样本统计:")
        print(f"总记录数: {len(positive_counts_per_record)}")
        print(f"总正样本数: {sum(positive_counts_per_record)}")
        if positive_counts_per_record:
            print(f"平均每记录正样本数: {np.mean(positive_counts_per_record):.2f}")
            print(f"最大正样本数: {max(positive_counts_per_record)}")
            print(f"最小正样本数: {min(positive_counts_per_record)}")


if __name__ == "__main__":
    # 先分析大的QA数据集
    analyze_qa_dataset()

    # 再分析小的数据集
    analyze_smaller_dataset()