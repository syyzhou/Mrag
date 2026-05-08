import pandas as pd
import json
import argparse
import io
import pandas as pd
import json
import sys
import math
from tqdm import tqdm




def generate_qa_pairs(main_data, historical_data=None, args=None):
    # Sort the dataframe by UserId, pseudo_session_trajectory_id, and timestamp
    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])

    # List to store the QA pairs
    qa_pairs = []

    # Iterate over each user
    for user in tqdm(main_data['UserId'].unique()):
        user_data = main_data[main_data['UserId'] == user]
        all_trajectories = user_data.sort_values('UTCTimeOffsetEpoch').groupby('pseudo_session_trajectory_id')
        trajectory_list = [(traj_id, traj_data) for traj_id, traj_data in all_trajectories]

        # Process each trajectory after the first (to ensure we have history)
        for i in range(0, len(trajectory_list)):
            current_traj_id, current_traj_data = trajectory_list[i]
            
            # Get up to 15 most recent historical trajectories
            historical_trajectories = trajectory_list[max(0, i-15):i]  # Last 15 or all available
            
            # Build question parts
            question_parts = [
                f"<question>: Given the user's historical check-ins and their current trajectory, predict which POI the user will visit next:"
                f" [Current trajectory of user {user} ]:"
            ]
            
            # Add current trajectory (excluding last POI)
            for _, row in current_traj_data.iloc[:-1].iterrows():
                question_parts.append(
                    f"time: {row['UTCTimeOffset']}, POI id: {row['PoiId']}, category name: {row['PoiCategoryName']}; "
                    # f"stay duration: {row['stay_duration']} minutes, transition distance: {row['transition_distance']} meters,"
                )
            
            # Add historical trajectories
            if historical_trajectories:
                question_parts.append(f"[Historical check-in records of user {user}]:")
                
                for hist_traj_id, hist_traj_data in historical_trajectories:
                    # question_parts.append(f"[Sequence from {hist_traj_data['UTCTimeOffset'].iloc[0].split()[0]}]:")
                    for _, row in hist_traj_data.iterrows():
                        question_parts.append(
                            f"time: {row['UTCTimeOffset']}, POI id: {row['PoiId']}, category name: {row['PoiCategoryName']}; "
                            # f"stay duration: {row['stay_duration']} minutes, transition distance: {row['transition_distance']} meters,"
                        )
            
            # Join question parts and add prediction query
            question = " ".join(question_parts)
            value = {'nyc': 4981, 'tky': 7833, 'ca': 9690}[args.dataset_name]
            question += (
                f" Given the above, At {current_traj_data.iloc[-1]['UTCTimeOffset']}, "
                f"Which POI id will user {user} visit? "
                f"Note that POI id is an integer in the range from 0 to {value}."
            )
            
            # Create answer in your specified format
            answer = (
                f"<answer>: At {user_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id "
                f"{user_data.iloc[-1]['PoiId']}.category name {user_data.iloc[-1]['PoiCategoryName']}."
            )
            qa_pairs.append((question, answer))
    return qa_pairs

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process dataset names.")

    # Add an argument for the dataset name
    parser.add_argument("--dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")

    # Parse the arguments
    args = parser.parse_args()

    # Your processing code here
    print(f"Processing dataset: {args.dataset_name}")
    path = f'../datasets/{args.dataset_name}/preprocessed/'
    # Read the data
    train_data = pd.read_csv(f'{path}train_sample_with_duration_and_distance.csv')
    test_data = pd.read_csv(f'{path}test_sample_with_traj_with_duration_and_distance.csv')
    # kqt1 = jload(f'{path}train_key_top200.json')
    # kqt2 = jload(f'{path}test_key_top200.json')
    
    # kqt1 = jload(f'{path}train_key_top50.json')
    # kqt2 = jload(f'{path}test_key_top50.json')
    # Generate the QA pairs
    qa_pairs_train = generate_qa_pairs(train_data, historical_data=train_data, args=args)
    # qa_pairs_test = generate_qa_pairs(test_data, historical_data=train_data, args=args)

    # Save the train QA pairs in JSON formatd
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
# 改成50条历史数据和300条
    with open(f'{path}train_qa_pairs_kqt_structed.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)

    qa_pairs_test = generate_qa_pairs(test_data, historical_data=train_data, args=args)
    # # Save the test QA pairs in TXT format
    with open(f'{path}test_qa_pairs_kqt_structed.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')


if __name__ == "__main__":
    main()

