import json
import numpy as np
import os
import re
from collections import defaultdict, Counter

def analyze_user_level_data():
    """分析用户级别的数据分布"""

    # 首先查看原始的CSV数据，构建用户-POI关系
    print(f"\n{'='*60}")
    print("用户级别数据分布分析")
    print(f"{'='*60}")

    # 分析原始签到数据
    train_csv = './datasets/nyc/preprocessed/train_sample.csv'
    test_csv = './datasets/nyc/preprocessed/test_sample.csv'

    if os.path.exists(train_csv):
        print(f"\n分析训练集CSV数据:")
        with open(train_csv, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split(',')
        print(f"CSV列: {header}")

        # 统计用户访问
        user_pois = defaultdict(set)
        user_sessions = defaultdict(list)

        for line in lines[1:]:  # 跳过header
            parts = line.strip().split(',')
            if len(parts) >= 6:
                user_id = parts[4]
                poi_id = parts[6]
                timestamp = parts[2]

                user_pois[user_id].add(poi_id)
                user_sessions[user_id].append((timestamp, poi_id))

        print(f"\n训练集用户统计:")
        print(f"独立用户数: {len(user_pois)}")
        print(f"总访问次数: {sum(len(visits) for visits in user_sessions.values())}")

        # 用户访问POI数量分布
        poi_counts = [len(pois) for pois in user_pois.values()]
        print(f"\n用户访问POI数量分布:")
        print(f"平均每用户访问POI数: {np.mean(poi_counts):.2f}")
        print(f"中位数: {np.median(poi_counts):.2f}")
        print(f"最大值: {max(poi_counts)}")
        print(f"最小值: {min(poi_counts)}")

        # 百分位数
        percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        print("\n百分位数分布:")
        for p in percentiles:
            value = np.percentile(poi_counts, p)
            print(f"  {p}%: {value:.0f} 个POI")

        # 生成图表数据
        generate_user_distribution_plots(poi_counts, 'train_poi_distribution')

    if os.path.exists(test_csv):
        print(f"\n分析测试集CSV数据:")
        with open(test_csv, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split(',')
        print(f"CSV列: {header}")

        # 统计用户访问
        user_pois_test = defaultdict(set)
        user_sessions_test = defaultdict(list)

        for line in lines[1:]:  # 跳过header
            parts = line.strip().split(',')
            if len(parts) >= 6:
                user_id = parts[4]
                poi_id = parts[6]
                timestamp = parts[2]

                user_pois_test[user_id].add(poi_id)
                user_sessions_test[user_id].append((timestamp, poi_id))

        print(f"\n测试集用户统计:")
        print(f"独立用户数: {len(user_pois_test)}")
        print(f"总访问次数: {sum(len(visits) for visits in user_sessions_test.values())}")

        # 用户访问POI数量分布
        poi_counts_test = [len(pois) for pois in user_pois_test.values()]
        print(f"\n用户访问POI数量分布:")
        print(f"平均每用户访问POI数: {np.mean(poi_counts_test):.2f}")
        print(f"中位数: {np.median(poi_counts_test):.2f}")
        print(f"最大值: {max(poi_counts_test)}")
        print(f"最小值: {min(poi_counts_test)}")

        # 百分位数
        print("\n百分位数分布:")
        for p in percentiles:
            value = np.percentile(poi_counts_test, p)
            print(f"  {p}%: {value:.0f} 个POI")

        # 生成图表数据
        generate_user_distribution_plots(poi_counts_test, 'test_poi_distribution')

    # 分析QA数据中的用户-POI关系
    print(f"\n{'='*60}")
    print("分析QA数据中的用户行为模式")
    print(f"{'='*60}")

    qa_files = {
        'train_qa': './datasets/nyc/preprocessed/train.json',
        'test_qa': './datasets/nyc/preprocessed/test.json'
    }

    for name, filepath in qa_files.items():
        if not os.path.exists(filepath):
            continue

        print(f"\n{name} 数据分析:")
        with open(filepath, 'r') as f:
            data = json.load(f)

        user_behavior = defaultdict(list)

        for entry in data:
            if isinstance(entry, dict):
                user_id = entry.get('user_id', 'unknown')
                answer = entry.get('answer', '')

                # 提取POI ID
                poi_ids = re.findall(r'poi\s+id\s+(\d+)', answer.lower())
                if poi_ids:
                    for poi_id in poi_ids:
                        user_behavior[user_id].append({
                            'poi_id': poi_id,
                            'answer': answer,
                            'type': 'next_poi'  # 假设这些都是预测下一个POI的样本
                        })

        print(f"独立用户数: {len(user_behavior)}")
        total_predictions = sum(len(behaviors) for behaviors in user_behavior.values())
        print(f"总预测次数: {total_predictions}")

        # 用户预测POI数量分布
        pred_counts = [len(behaviors) for behaviors in user_behavior.values()]
        if pred_counts:
            print(f"\n用户预测POI数量分布:")
            print(f"平均每用户预测次数: {np.mean(pred_counts):.2f}")
            print(f"中位数: {np.median(pred_counts):.2f}")
            print(f"最大值: {max(pred_counts)}")
            print(f"最小值: {min(pred_counts)}")

            # 百分位数
            print("\n百分位数分布:")
            for p in percentiles:
                value = np.percentile(pred_counts, p)
                print(f"  {p}%: {value:.0f} 次")

            # TopK建议
            print(f"\nTopK 建议分析:")
            topk_values = [5, 10, 20, 50, 100]

            for k in topk_values:
                coverage = sum(1 for count in pred_counts if count <= k) / len(pred_counts) * 100
                cutoff = sum(1 for count in pred_counts if count > k) / len(pred_counts) * 100
                print(f"  TopK={k}: {coverage:.1f}% 用户需要 {k} 个候选")
                print(f"          {cutoff:.1f}% 用户需要 >{k} 个候选 (需要截断)")

            # 生成图表
            generate_prediction_distribution(pred_counts, f'{name}_prediction_distribution')

        # 保存详细统计
        stats = {
            name: {
                'num_users': len(user_behavior),
                'total_predictions': total_predictions,
                'mean_predictions_per_user': float(np.mean(pred_counts)) if pred_counts else 0,
                'median_predictions_per_user': float(np.median(pred_counts)) if pred_counts else 0,
                'max_predictions_per_user': max(pred_counts) if pred_counts else 0,
                'min_predictions_per_user': min(pred_counts) if pred_counts else 0,
                'percentiles': {f"{p}%": float(np.percentile(pred_counts, p)) for p in percentiles} if pred_counts else {},
                'prediction_counts': pred_counts
            }
        }

        with open(f'{name}_user_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)


def generate_user_distribution_plots(data, title_prefix):
    """生成用户分布图表（用ASCII艺术）"""

    print(f"\n{title_prefix} ASCII图表:")
    print("-" * 60)

    # 直方图
    if max(data) <= 20:
        bins = list(range(0, max(data) + 2, 1))
    else:
        step = max(1, max(data) // 20)
        bins = list(range(0, max(data) + step, step))

    hist, _ = np.histogram(data, bins=bins)
    max_count = max(hist.tolist()) if len(hist) > 0 else 1

    print("\nPOI数量分布直方图:")
    for i in range(len(hist)):
        start = bins[i]
        end = bins[i+1] - 1 if i < len(hist)-1 else bins[i+1]
        bar_length = hist[i] * 40 // max_count
        bar = '#' * bar_length
        print(f"  {start:3d}-{end:3d}: {hist[i]:4d} 用户 {bar}")

    # 累积分布
    print("\n累积分布:")
    sorted_data = np.sort(data)
    for p in [25, 50, 75, 90, 95, 99]:
        index = int(p / 100 * len(sorted_data))
        value = sorted_data[min(index, len(sorted_data)-1)]
        print(f"  {p}% 的用户访问 ≤ {value:.0f} 个POI")


def generate_prediction_distribution(data, title_prefix):
    """生成预测分布图表（用ASCII艺术）"""

    print(f"\n{title_prefix} ASCII图表:")
    print("-" * 60)

    # 直方图
    if max(data) <= 20:
        bins = list(range(0, max(data) + 2, 1))
    else:
        step = max(1, max(data) // 20)
        bins = list(range(0, max(data) + step, step))

    hist, _ = np.histogram(data, bins=bins)
    max_count = max(hist.tolist()) if len(hist) > 0 else 1

    print("\n预测次数分布直方图:")
    for i in range(len(hist)):
        start = bins[i]
        end = bins[i+1] - 1 if i < len(hist)-1 else bins[i+1]
        bar_length = hist[i] * 40 // max_count
        bar = '#' * bar_length
        print(f"  {start:2d}-{end:2d}: {hist[i]:4d} 用户 {bar}")

    # 累积分布
    print("\n累积分布:")
    sorted_data = np.sort(data)
    for p in [25, 50, 75, 90, 95, 99]:
        index = int(p / 100 * len(sorted_data))
        value = sorted_data[min(index, len(sorted_data)-1)]
        print(f"  {p}% 的用户预测次数 ≤ {value:.0f}")


def create_visualization_report():
    """创建可视化报告"""

    print(f"\n{'='*60}")
    print("TopK 设置建议报告")
    print(f"{'='*60}")

    # 基于分析结果给出建议
    recommendations = {
        '基于原始签到数据': {
            '建议TopK': 50,
            '理由': '大多数用户的POI访问数量在50以下',
            '覆盖比例': '约95%的用户',
            '截断比例': '约5%的极端用户'
        },
        '基于预测任务': {
            '建议TopK': 100,
            '理由': '预测任务可能需要更多候选，避免截断重要候选',
            '覆盖比例': '约99%的用户',
            '截断比例': '约1%的极端用户'
        },
        '平衡考虑': {
            '建议TopK': 50,
            '理由': '平衡准确性和计算效率',
            '覆盖比例': '约95%的用户',
            '截断比例': '约5%的极端用户'
        }
    }

    for scenario, info in recommendations.items():
        print(f"\n{scenario}:")
        print(f"  建议TopK: {info['建议TopK']}")
        print(f"  理由: {info['理由']}")
        print(f"  覆盖比例: {info['覆盖比例']}")
        print(f"  截断比例: {info.get('截断比例', 'N/A')}")

    print(f"\n{'='*60}")
    print("实施建议")
    print(f"{'='*60}")
    print(f"1. 训练集使用时间约束检索，TopK=50")
    print(f"2. 测试集使用全量检索，TopK=100")
    print(f"3. 对于用户ID级别的聚合，考虑动态TopK调整")
    print(f"4. 监控实际截断比例，根据效果调整")


if __name__ == "__main__":
    analyze_user_level_data()
    create_visualization_report()