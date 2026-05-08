import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm
import torch

# 导入数据加载模块
from poi_data_loader import load_full_dataset
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
    print("\n" + "="*50)
    print("数据集统计信息")
    print("="*50)
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

    # 绘制分布图
    plt.figure(figsize=(15, 10))

    # 1. 训练集正样本分布
    plt.subplot(2, 2, 1)
    plt.hist(train_positive_counts, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(train_positive_counts), color='red', linestyle='--',
                label=f'平均值: {np.mean(train_positive_counts):.2f}')
    plt.axvline(np.median(train_positive_counts), color='green', linestyle='--',
                label=f'中位数: {np.median(train_positive_counts):.2f}')
    plt.xlabel('每用户正样本数')
    plt.ylabel('用户数')
    plt.title('训练集正样本分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 测试集正样本分布
    plt.subplot(2, 2, 2)
    plt.hist(test_positive_counts, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(np.mean(test_positive_counts), color='red', linestyle='--',
                label=f'平均值: {np.mean(test_positive_counts):.2f}')
    plt.axvline(np.median(test_positive_counts), color='green', linestyle='--',
                label=f'中位数: {np.median(test_positive_counts):.2f}')
    plt.xlabel('每用户正样本数')
    plt.ylabel('用户数')
    plt.title('测试集正样本分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 累积分布
    plt.subplot(2, 2, 3)
    train_sorted = np.sort(train_positive_counts)
    test_sorted = np.sort(test_positive_counts)

    plt.plot(range(len(train_sorted)), train_sorted, 'b-', label='训练集', linewidth=2)
    plt.plot(range(len(test_sorted)), test_sorted, 'orange', label='测试集', linewidth=2)
    plt.xlabel('用户排序（按正样本数升序）')
    plt.ylabel('正样本数')
    plt.title('正样本数累积分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 百分位数分布
    plt.subplot(2, 2, 4)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    train_percentiles = [np.percentile(train_positive_counts, p) for p in percentiles]
    test_percentiles = [np.percentile(test_positive_counts, p) for p in percentiles]

    x = np.arange(len(percentiles))
    width = 0.35
    plt.bar(x - width/2, train_percentiles, width, label='训练集', alpha=0.7, color='blue')
    plt.bar(x + width/2, test_percentiles, width, label='测试集', alpha=0.7, color='orange')

    plt.xlabel('百分位数')
    plt.ylabel('正样本数')
    plt.title('正样本数百分位数分布')
    plt.xticks(x, [f'{p}%' for p in percentiles])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("\n分布图已保存为: data_distribution_analysis.png")

    # 生成统计报告
    print("\n" + "="*50)
    print("TopK 建议分析")
    print("="*50)

    # 考虑不同topk设置
    topk_values = [5, 10, 20, 50, 100]

    for k in topk_values:
        train_coverage = sum(1 for count in train_positive_counts if count <= k) / len(train_positive_counts) * 100
        test_coverage = sum(1 for count in test_positive_counts if count <= k) / len(test_positive_counts) * 100

        print(f"\nTopK = {k}:")
        print(f"  训练集覆盖用户比例: {train_coverage:.1f}%")
        print(f"  测试集覆盖用户比例: {test_coverage:.1f}%")
        print(f"  训练集需截断用户比例: {100-train_coverage:.1f}%")
        print(f"  测试集需截断用户比例: {100-test_coverage:.1f}%")

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

    print("\n详细统计结果已保存为: data_statistics.json")

    return train_positive_counts, test_positive_counts


if __name__ == "__main__":
    analyze_positive_samples_distribution()