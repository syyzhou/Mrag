import pandas as pd
import json
import argparse
import io
from tqdm import tqdm
from datetime import datetime

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def get_time_diff_string(current_time, history_time):
    """
    计算时间差并返回可读字符串 (e.g., '3 days ago', '2 weeks ago', '1 months ago')
    """
    diff = current_time - history_time
    days = diff.days

    if days < 7:
        return f"{max(0, days)} days ago"
    elif days < 30:
        weeks = days // 7
        return f"{weeks} weeks ago"
    elif days < 365:
        months = days // 30
        return f"{months} months ago"
    else:
        years = days // 365
        return f"{years} years ago"

def build_user_trajectory_list(df):
    """
    预处理：将 DataFrame 按用户和轨迹ID分组，转成按时间顺序的轨迹列表。
    返回结构: { user_id: [traj_df_1, traj_df_2, ...] }
    """
    user_traj_dict = {}
    
    # 按用户分组
    for user, user_df in tqdm(df.groupby('UserId'), desc="Indexing trajectories"):
        # 1. 先按时间排序
        user_df = user_df.sort_values(by='UTCTimeOffsetEpoch').reset_index(drop=True)
        
        # 2. 按轨迹 ID 分组（保持时间顺序）
        traj_groups = [traj_data for _, traj_data in user_df.groupby('pseudo_session_trajectory_id', sort=False)]
        
        # 3. 按每条轨迹的首条时间排序整个轨迹列表
        traj_groups.sort(key=lambda x: x['UTCTimeOffsetEpoch'].iloc[0])
        
        user_traj_dict[user] = traj_groups
        
    return user_traj_dict


def generate_qa_pairs(target_data_dict, train_history_dict=None, mode='train', args=None):
    """
    target_data_dict: 当前要处理的数据字典 {user: [traj_list]}
    train_history_dict: 训练集的数据字典 (如果传 None，则不使用外部历史)
    """
    qa_pairs = []

    # 总窗口大小 = 当前轨迹 + 历史轨迹
    TOTAL_WINDOW_SIZE = 30

    for user in tqdm(target_data_dict.keys(), desc=f"Generating {mode} pairs"):
        target_trajs = target_data_dict[user]
        # 如果 train_history_dict 为 None，这里就是空列表，实现了测试集不看训练集历史
        train_trajs = train_history_dict.get(user, []) if train_history_dict else []

        for i, current_traj_data in enumerate(target_trajs):
            # ---------------------------
            # 1. 构建历史轨迹
            # ---------------------------
            # 获取当前轨迹的开始时间对象 (用于计算时间差)
            curr_start_time_obj = pd.to_datetime(current_traj_data['UTCTimeOffset'].iloc[0])

            if mode == 'train':
                # 训练集只取当前轨迹前的轨迹
                history_trajs = target_trajs[max(0, i - TOTAL_WINDOW_SIZE + 1): i]
            else:
                # 测试集逻辑修改：
                # 现在的逻辑是：full_history = 外部传入的train历史(可能是空) + 当前target之前的历史
                # 因为 main 函数里我们会传 train_history_dict=None，所以这里实际上只包含 test 自己的历史
                full_history = train_trajs + target_trajs[:i]
                history_trajs = full_history[-TOTAL_WINDOW_SIZE:]  

            # ---------------------------
            # 2. 当前轨迹内部按时间排序
            # ---------------------------
            current_traj_data = current_traj_data.sort_values('UTCTimeOffsetEpoch').reset_index(drop=True)
            history_trajs = [ht.sort_values('UTCTimeOffsetEpoch').reset_index(drop=True) for ht in history_trajs]

            # ---------------------------
            # 3. 构造 Question
            # ---------------------------
            question_parts = [f"<question>: The following data contains check-in sequences of user {user}."]
            question_parts.append("[Current trajectory's check-in sequence]:")

            # 当前轨迹除最后一个点（预测目标）
            for _, row in current_traj_data.iloc[:-1].iterrows():
                row_time = pd.to_datetime(row['UTCTimeOffset']).strftime('%Y-%m-%d %H:%M:%S')
                question_parts.append(
                    f"At {row_time}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} with Category id {row['PoiCategoryId']}."
                )

            # 历史轨迹
            if history_trajs:
                question_parts.append("[Historical check-in sequences]:")
                for hist_traj_data in history_trajs:
                    # 获取该条历史轨迹的起始时间对象
                    hist_start_time_obj = pd.to_datetime(hist_traj_data['UTCTimeOffset'].iloc[0])
                    start_time_str = hist_start_time_obj.strftime('%Y-%m-%d')
                    
                    # 【修改点 1】计算时间差字符串
                    time_ago_str = get_time_diff_string(curr_start_time_obj, hist_start_time_obj)

                    # 【修改点 2】将时间差加入 Prompt
                    # question_parts.append(f"[Sequence from {start_time_str} ({time_ago_str})]:")
                    question_parts.append(f"[Sequence from {start_time_str}]:")
                    
                    for _, row in hist_traj_data.iterrows():
                        row_time = pd.to_datetime(row['UTCTimeOffset']).strftime('%Y-%m-%d %H:%M:%S')
                        question_parts.append(
                            f"At {row_time}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} with Category id {row['PoiCategoryId']}."
                        )

            # ---------------------------
            # 4. 提问 Trigger
            # ---------------------------
            target_row = current_traj_data.iloc[-1]
            target_time = pd.to_datetime(target_row['UTCTimeOffset']).strftime('%Y-%m-%d %H:%M:%S')
            target_poi = target_row['PoiId']
            limit_val = {'nyc': 4981, 'tky': 7833, 'ca': 9690}.get(args.dataset_name, 10000)

            question_parts.append(
                f"Given the data, At {target_time}, Which POI id will user {user} visit? "
                f"Note that POI id is an integer in the range from 0 to {limit_val}."
            )

            question_str = " ".join(question_parts)
            answer_str = f"<answer>: At {target_time}, user {user} will visit POI id {target_poi}.{target_row['PoiCategoryName']}."

            qa_pairs.append((question_str, answer_str))

    return qa_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='nyc', choices=['ca', 'nyc', 'tky'])
    args = parser.parse_args()

    print(f"Processing dataset: {args.dataset_name}")
    path = f'../datasets/{args.dataset_name}/preprocessed/'
    
    # 1. 读取 CSV 数据
    print("Loading data...")
    train_df = pd.read_csv(f'{path}train_sample.csv') 
    test_df = pd.read_csv(f'{path}test_sample_with_traj.csv')

    # 2. 预处理：将 dataframe 转为轨迹列表字典 {user: [traj1, traj2...]}
    print("Building Trajectory Indexes...")
    train_traj_dict = build_user_trajectory_list(train_df)
    test_traj_dict = build_user_trajectory_list(test_df)

    # 3. 生成 QA Pairs
    
    # A. 训练集生成
    print("Generating Train Pairs...")
    qa_pairs_train = generate_qa_pairs(
        target_data_dict=train_traj_dict, 
        train_history_dict=None, 
        mode='train', 
        args=args
    )

    # B. 测试集生成
    # 【修改点 3】这里 train_history_dict 改为 None
    # 这样测试集只使用 test_traj_dict 自身进行滑动窗口，不拼接 train 的数据
    print("Generating Test Pairs (Independent sliding window)...")
    qa_pairs_test = generate_qa_pairs(
        target_data_dict=test_traj_dict, 
        train_history_dict=None, # 修改为 None，不补充训练集历史
        mode='test', 
        args=args
    )

    # 4. 保存结果
    print(f"Saving {len(qa_pairs_train)} train samples...")
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt_15.json', 'w') as f:
        json.dump(qa_dict_train, f, indent=2)

    print(f"Saving {len(qa_pairs_test)} test samples...")
    with open(f'{path}test_qa_pairs_kqt_15.txt', 'w') as f:
        for q, a in qa_pairs_test:
            # 格式根据你的需求，这里直接拼接
            f.write(q + " " + a + '\n')
    # qa_dict_test = [{"question": q, "answer": a} for q, a in qa_pairs_test]
    # with open(f'{path}validate_qa_pairs_kqt.json', 'w') as f:
    #     json.dump(qa_dict_test, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()