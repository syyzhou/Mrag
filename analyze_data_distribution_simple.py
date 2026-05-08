import os
import json
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# 导入数据加载模块
import sys
sys.path.append('./rag')
from poi_data_loader import load_full_dataset

# 导入时间桶管理器
sys.path.append('./')
from transition_graph import TimeBucketManager, build_transition_graph


def analyze_positive_samples_distribution():
    """分析训练集和测试集中的正样本分布"""

    print("加载数据...")
    # 加载数据
    dataset_path = "data/poi_dataset.json"
    full_dataset = load_full_dataset(dataset_path)

    print("构建时间桶管理器...")
    # 构建时间桶管理器
    time_bucket_manager = TimeBucketManager()
    buckets = time_bucket_manager.create_time_buckets()

    print("构建时间图...")
    # 构建时间图（用于训练集的时间约束）
    time_graph = build_transition_graph(buckets)

    # 统计数据
    train_positive_counts = []
    test_positive_counts = []
    train_users = set()
    test_users = set()

    print("分析训练集数据...")
    # 分析训练集
    for user_id, sessions in full_dataset['train'].items():
        train_users.add(user_id)
        total_positives = 0

        for session in sessions:
            positives = []
            for item in session['items']:
                if item.get('label', 0) == 1:  # 正样本
                    positives.append(item)

            total_positives += len(positives)

        train_positive_counts.append(total_positives)

    print("分析测试集数据...")
    # 分析测试集
    for user_id, sessions in full_dataset['test'].items():
        test_users.add(user_id)
        total_positives = 0

        for session in sessions:
            positives = []
            for item in session['items']:
                if item.get('label', 0) == 1:  # 正样本
                    positives.append(item)

            total_positives += len(positives)

        test_positive_counts.append(total_positives)

    # 打印基本统计信息
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    print(f"训练集用户数: {len(train_users)}")
    print(f"测试集用户数: {len(test_users)}")
    print(f"训练集总正样本数: {sum(train_positive_counts)}")
    print(f"测试集总正样本数: {sum(test_positive_counts)}")
    print(f"训练集平均每用户正样本数: {np.mean(train_positive_counts):.2f}")
    print(f"测试集平均每用户正样本数: {np.mean(test_positive_counts):.2f}")
    print(f"训练集最大正样本数: {max(train_positive_counts)}")
    print(f"测试集最大正样本数: {max(test_positive_counts)}")
    print(f"训练集最小正样本数: {min(train_positive_counts)}")
    print(f"测试集最小正样本数: {min(test_positive_counts)}")

    # 打印百分位数
    print("\n" + "="*60)
    print("正样本数百分位数分布")
    print("="*60)
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]

    print("\n训练集:")
    for p in percentiles:
        value = np.percentile(train_positive_counts, p)
        print(f"  {p}%: {value:.0f} 个正样本")

    print("\n测试集:")
    for p in percentiles:
        value = np.percentile(test_positive_counts, p)
        print(f"  {p}%: {value:.0f} 个正样本")

    # 打印直方图分布
    print("\n" + "="*60)
    print("正样本数分布直方图")
    print("="*60)

    # 计算直方图 bins
    def print_histogram(data, title):
        print(f"\n{title}:")

        # 计算合适的bins
        min_val = min(data)
        max_val = max(data)
        if max_val <= 20:
            bins = list(range(0, max_val + 2, 1))
        else:
            step = max(1, (max_val - min_val) // 20)
            bins = list(range(0, max_val + step, step))

        if bins[-1] < max_val:
            bins.append(max_val + 1)

        # 统计每个bin的数量
        hist = np.histogram(data, bins=bins)[0]

        # 打印直方图
        max_count = max(hist) if hist else 1
        for i in range(len(hist)):
            start = bins[i]
            end = bins[i+1] - 1
            bar = '#' * (hist[i] * 50 // max_count)
            count = hist[i]
            print(f"  {start:3d}-{end:3d}: {count:4d} 用户 {bar}")

    print_histogram(train_positive_counts, "训练集正样本分布")
    print_histogram(test_positive_counts, "测试集正样本分布")

    # 生成TopK建议分析
    print("\n" + "="*60)
    print("TopK 建议分析")
    print("="*60)

    # 考虑不同topk设置
    topk_values = [5, 10, 20, 30, 50, 100, 150, 200]

    print("\n详细分析:")
    print("-" * 80)
    print(f"{'TopK':<5} {'训练集覆盖用户':<18} {'测试集覆盖用户':<18} {'训练集截断率':<12} {'测试集截断率':<12}")
    print("-" * 80)

    for k in topk_values:
        train_coverage = sum(1 for count in train_positive_counts if count <= k) / len(train_positive_counts) * 100
        test_coverage = sum(1 for count in test_positive_counts if count <= k) / len(test_positive_counts) * 100

        print(f"{k:<5} {train_coverage:<17.1f}% {test_coverage:<17.1f}% {100-train_coverage:<11.1f}% {100-test_coverage:<11.1f}%")

    # 分析极端情况
    print("\n" + "="*60)
    print("极端情况分析")
    print("="*60)

    # 找出正样本数特别多的用户
    print("\n正样本数最多的10个训练集用户:")
    sorted_train = sorted([(count, i) for i, count in enumerate(train_positive_counts)], reverse=True)
    for i, (count, idx) in enumerate(sorted_train[:10]):
        print(f"  第{i+1}位: {count} 个正样本")

    print("\n正样本数最多的10个测试集用户:")
    sorted_test = sorted([(count, i) for i, count in enumerate(test_positive_counts)], reverse=True)
    for i, (count, idx) in enumerate(sorted_test[:10]):
        print(f"  第{i+1}位: {count} 个正样本")

    # 保存详细统计结果
    stats = {
        'train': {
            'num_users': len(train_users),
            'total_positives': sum(train_positive_counts),
            'mean_positives': float(np.mean(train_positive_counts)),
            'median_positives': float(np.median(train_positive_counts)),
            'max_positives': int(max(train_positive_counts)),
            'min_positives': int(min(train_positive_counts)),
            'std_positives': float(np.std(train_positive_counts)),
            'counts': train_positive_counts
        },
        'test': {
            'num_users': len(test_users),
            'total_positives': sum(test_positive_counts),
            'mean_positives': float(np.mean(test_positive_counts)),
            'median_positives': float(np.median(test_positive_counts)),
            'max_positives': int(max(test_positive_counts)),
            'min_positives': int(min(test_positive_counts)),
            'std_positives': float(np.std(test_positive_counts)),
            'counts': test_positive_counts
        }
    }

    with open('data_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n详细统计结果已保存为: data_statistics.json")

    return train_positive_counts, test_positive_counts


if __name__ == "__main__":
    analyze_positive_samples_distribution()