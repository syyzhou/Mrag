import json
import numpy as np
import os
from collections import defaultdict, Counter

def analyze_dataset_structure():
    """直接分析数据集结构"""

    # 查找数据文件
    data_files = {
        'train': './datasets/nyc/preprocessed/train.json',
        'test': './datasets/nyc/preprocessed/test.json'
    }

    datasets = {}

    # 读取数据
    for name, filepath in data_files.items():
        if not os.path.exists(filepath):
            print(f"警告: {filepath} 不存在")
            continue

        print(f"\n读取 {name} 数据...")
        with open(filepath, 'r') as f:
            data = json.load(f)

        # 分析数据结构
        print(f"\n{name} 数据结构分析:")
        print(f"总条目数: {len(data)}")
        print(f"数据类型: {type(data)}")

        # 统计正样本数量
        positive_counts = []
        total_positives = 0
        total_items = 0

        # 看前几个样本的结构
        if isinstance(data, list):
            # 数据是列表格式
            for i, entry in enumerate(data[:5]):  # 只看前5个条目
                print(f"\n条目 {i+1}:")
                print(f"  类型: {type(entry)}")
                print(f"  内容: {json.dumps(entry, indent=2, ensure_ascii=False)[:200]}...")

                if isinstance(entry, dict):
                    user_positives = 0
                    for key, value in entry.items():
                        print(f"    {key}: {type(value)}")
                        if isinstance(value, list):
                            session_count = len(value)
                            print(f"    会话数: {session_count}")

                            for session in value:
                                if isinstance(session, dict) and 'items' in session:
                                    for item in session['items']:
                                        if isinstance(item, dict) and item.get('label', 0) == 1:
                                            user_positives += 1
                                            total_positives += 1
                                        total_items += 1

                    positive_counts.append(user_positives)
                    print(f"  正样本数: {user_positives}")
        else:
            # 数据可能是字典格式
            for i, (user_id, sessions) in enumerate(data.items()):
                if i >= 5:  # 只看前5个用户
                    break

                user_positives = 0
                session_count = len(sessions)

                for session in sessions:
                    # 检查session结构
                    if 'items' in session:
                        for item in session['items']:
                            if isinstance(item, dict) and item.get('label', 0) == 1:
                                user_positives += 1
                                total_positives += 1
                            total_items += 1

                positive_counts.append(user_positives)

                print(f"\n用户 {user_id}:")
                print(f"  会话数: {session_count}")
                print(f"  正样本数: {user_positives}")

                if i == 0:  # 详细显示第一个用户的第一会话
                    print(f"  第一个会话详情:")
                    first_session = sessions[0]
                    for j, item in enumerate(first_session['items'][:10]):  # 只显示前10个
                        label = item.get('label', 0)
                        print(f"    {j+1}. POI {item.get('poi_id', 'unknown')} - Label: {label}")

        datasets[name] = {
            'data': data,
            'total_positives': total_positives,
            'positive_counts': positive_counts,
            'total_items': total_items
        }

        print(f"\n{name} 集合统计:")
        print(f"总正样本数: {total_positives}")
        print(f"总项目数: {total_items}")
        if positive_counts:
            print(f"平均每用户正样本数: {np.mean(positive_counts):.2f}")
            print(f"最大正样本数: {max(positive_counts)}")
            print(f"最小正样本数: {min(positive_counts)}")

    # 更详细的分析所有用户
    print("\n" + "="*60)
    print("完整数据集分析")
    print("="*60)

    for name, dataset in datasets.items():
        data = dataset['data']
        all_positive_counts = []

        print(f"\n{name} 集合详细分析:")

        # 根据数据类型进行处理
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    user_positives = 0
                    # 提取用户ID
                    user_id = str(entry.get('user_id', hash(str(entry))))

                    # 查找包含标签的信息
                    for key, value in entry.items():
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and item.get('label', 0) == 1:
                                    user_positives += 1

                    all_positive_counts.append(user_positives)
        else:
            for user_id, sessions in data.items():
                user_positives = 0
                for session in sessions:
                    if 'items' in session:
                        for item in session['items']:
                            if isinstance(item, dict) and item.get('label', 0) == 1:
                                user_positives += 1

                all_positive_counts.append(user_positives)

        # 打印统计信息
        print(f"用户总数: {len(data)}")
        print(f"总正样本数: {sum(all_positive_counts)}")
        print(f"平均每用户正样本数: {np.mean(all_positive_counts):.2f}")
        print(f"中位数: {np.median(all_positive_counts):.2f}")
        print(f"最大值: {max(all_positive_counts)}")
        print(f"最小值: {min(all_positive_counts)}")

        # 打印百分位数
        percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        print("\n百分位数分布:")
        for p in percentiles:
            value = np.percentile(all_positive_counts, p)
            print(f"  {p}%: {value:.0f} 个正样本")

        # 直方图
        print("\n分布直方图:")
        hist_data = all_positive_counts

        if max(hist_data) <= 20:
            bins = list(range(0, max(hist_data) + 2, 1))
        else:
            step = max(1, max(hist_data) // 20)
            bins = list(range(0, max(hist_data) + step, step))

        if bins[-1] < max(hist_data):
            bins.append(max(hist_data) + 1)

        hist, _ = np.histogram(hist_data, bins=bins)
        max_count = max(hist) if hist else 1

        for i in range(len(hist)):
            start = bins[i]
            end = bins[i+1] - 1
            bar_length = hist[i] * 40 // max_count
            bar = '#' * bar_length
            print(f"  {start:2d}-{end:2d}: {hist[i]:3d} 用户 {bar}")

        # TopK建议
        print("\nTopK 覆盖分析:")
        topk_values = [5, 10, 20, 50, 100]

        for k in topk_values:
            coverage = sum(1 for count in all_positive_counts if count <= k) / len(all_positive_counts) * 100
            print(f"  TopK={k}: {coverage:.1f}% 用户正样本数 ≤ {k}")

        # 保存结果
        result = {
            name: {
                'num_users': len(data),
                'total_positives': sum(all_positive_counts),
                'mean_positives': float(np.mean(all_positive_counts)),
                'median_positives': float(np.median(all_positive_counts)),
                'max_positives': int(max(all_positive_counts)),
                'min_positives': int(min(all_positive_counts)),
                'std_positives': float(np.std(all_positive_counts)),
                'counts': all_positive_counts
            }
        }

        with open(f'{name}_statistics.json', 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n详细统计结果已保存为: {name}_statistics.json")

    return datasets


if __name__ == "__main__":
    analyze_dataset_structure()