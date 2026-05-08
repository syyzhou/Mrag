import json
import numpy as np
import os
import re
from collections import defaultdict

def analyze_qa_samples():
    """分析QA数据集中每个样本的正样本数量"""

    # 数据集文件路径（与train_sem.py中一致）
    train_file = './datasets/nyc/preprocessed/train_qa_pairs_kqt.json'
    test_file = './datasets/nyc/preprocessed/test_qa_pairs_kqt.json'

    # 存储分析结果
    results = {}

    for name, filepath in [('训练集', train_file), ('测试集', test_file)]:
        print(f"\n{'='*60}")
        print(f"分析{name}: {filepath}")
        print(f"{'='*60}")

        if not os.path.exists(filepath):
            print(f"警告: 文件不存在 - {filepath}")
            continue

        # 加载数据
        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"总样本数: {len(data)}")

        # 统计每个样本的正样本数量
        sample_positive_counts = []
        sample_info = []

        # 用户级别的统计
        user_positive_counts = defaultdict(int)

        for i, sample in enumerate(data):
            if not isinstance(sample, dict):
                continue

            # 提取答案
            answer = sample.get('answer', '')

            # 从答案中提取POI ID
            # 尝试多种匹配模式
            poi_ids = []
            # 模式1: "poi id 123"
            poi_ids.extend(re.findall(r'poi\s+id\s+(\d+)', answer.lower()))
            # 模式2: "POI id 123"
            poi_ids.extend(re.findall(r'POI\s+id\s+(\d+)', answer.lower()))
            # 模式3: "poI id 123"
            poi_ids.extend(re.findall(r'poI\s+id\s+(\d+)', answer.lower()))
            # 模式4: 提取所有数字（作为POI ID候选）
            numbers = re.findall(r'\b(\d+)\b', answer)
            # 过滤掉明显不是POI ID的数字（如时间戳）
            filtered_numbers = []
            for num in numbers:
                # 过滤掉小于100的数字（可能是时间戳或其他ID）
                if len(num) >= 3 and num != '2012' and num != '2013':
                    filtered_numbers.append(num)
            poi_ids.extend(filtered_numbers)

            # 去重
            poi_ids = list(set(poi_ids))

            # 统计
            positive_count = len(poi_ids)
            sample_positive_counts.append(positive_count)

            # 用户统计
            user_id = sample.get('user_id', f'unknown_{i}')
            user_positive_counts[user_id] += positive_count

            # 记录样本信息（只记录前10个作为示例）
            if i < 10:
                sample_info.append({
                    'index': i,
                    'user_id': user_id,
                    'answer_preview': answer[:100] + '...' if len(answer) > 100 else answer,
                    'positive_count': positive_count,
                    'poi_ids': poi_ids[:5]  # 只显示前5个POI ID
                })

        # 打印样本信息示例
        print(f"\n前10个样本示例:")
        for info in sample_info:
            print(f"\n样本 {info['index']} (用户: {info['user_id']}):")
            print(f"  答案预览: {info['answer_preview']}")
            print(f"  正样本数: {info['positive_count']}")
            print(f"  POI IDs: {info['poi_ids']}")

        # 基本统计
        print(f"\n正样本数量统计:")
        print(f"总样本数: {len(data)}")
        print(f"总正样本数: {sum(sample_positive_counts)}")
        print(f"平均每样本正样本数: {np.mean(sample_positive_counts):.2f}")
        print(f"中位数: {np.median(sample_positive_counts):.2f}")
        print(f"最大值: {max(sample_positive_counts)}")
        print(f"最小值: {min(sample_positive_counts)}")

        # 百分位数分布
        percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        print(f"\n百分位数分布:")
        for p in percentiles:
            value = np.percentile(sample_positive_counts, p)
            print(f"  {p}%: {value:.0f} 个正样本")

        # 直方图分布
        print(f"\n正样本数量分布直方图:")
        hist_data = sample_positive_counts
        max_val = max(hist_data)
        if max_val <= 20:
            bins = list(range(0, max_val + 2, 1))
        else:
            step = max(1, max_val // 20)
            bins = list(range(0, max_val + step + 1, step))

        hist, _ = np.histogram(hist_data, bins=bins)
        max_count = max(hist) if len(hist) > 0 else 1

        for i in range(len(hist)):
            start = bins[i]
            end = bins[i+1] - 1 if i < len(hist)-1 else bins[i+1]
            bar_length = hist[i] * 40 // max_count
            bar = '#' * bar_length
            print(f"  {start:2d}-{end:2d}: {hist[i]:4d} 样本 {bar}")

        # TopK覆盖分析
        print(f"\nTopK 覆盖分析:")
        topk_values = [1, 2, 3, 5, 10, 20, 50, 100]

        print(f"{'TopK':<8} {'覆盖样本比例':<15} {'覆盖用户比例':<15} {'需截断比例':<12}")
        print("-" * 60)

        for k in topk_values:
            # 样本级别覆盖
            sample_coverage = sum(1 for count in sample_positive_counts if count <= k) / len(sample_positive_counts) * 100
            sample_cutoff = 100 - sample_coverage

            # 用户级别覆盖
            user_counts = list(user_positive_counts.values())
            user_coverage = sum(1 for count in user_counts if count <= k) / len(user_counts) * 100
            user_cutoff = 100 - user_coverage

            print(f"{k:<8} {sample_coverage:<14.1f}% {user_coverage:<14.1f}% {sample_cutoff:<11.1f}%")

        # 保存详细结果
        results[name] = {
            'file_path': filepath,
            'total_samples': len(data),
            'total_positives': sum(sample_positive_counts),
            'mean_positives_per_sample': float(np.mean(sample_positive_counts)),
            'median_positives_per_sample': float(np.median(sample_positive_counts)),
            'max_positives_per_sample': max(sample_positive_counts),
            'min_positives_per_sample': min(sample_positive_counts),
            'percentiles': {f"{p}%": float(np.percentile(sample_positive_counts, p)) for p in percentiles},
            'sample_positive_counts': sample_positive_counts,
            'user_positive_counts': dict(user_positive_counts)
        }

    # 保存结果到文件
    with open('qa_samples_analysis_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("分析完成！详细结果已保存到: qa_samples_analysis_result.json")
    print(f"{'='*60}")

    # 生成总结报告
    print("\n总结报告:")
    print("-" * 40)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  样本总数: {result['total_samples']}")
        print(f"  总正样本数: {result['total_positives']}")
        print(f"  平均每样本正样本数: {result['mean_positives_per_sample']:.2f}")
        print(f"  建议TopK: 50 (覆盖90%+的样本)")


if __name__ == "__main__":
    analyze_qa_samples()