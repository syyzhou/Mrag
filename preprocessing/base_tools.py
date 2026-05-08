from collections import defaultdict
import re
import pandas as pd
from geopy.distance import geodesic

def load_user_history(history_path):
    """Load user's historical trajectory data from the updated CSV file."""
    USER_HISTORY = {}
    try:
        # Read CSV file
        df = pd.read_csv(history_path)

        # Ensure 'UTCTimeOffset' is datetime type
        df['UTCTimeOffset'] = pd.to_datetime(df['UTCTimeOffset'])
        
        # Group by user_id
        grouped = df.groupby('UserId')

        for user_id, group in grouped:
            # Store each user's data as DataFrame
            USER_HISTORY[user_id] = group.reset_index(drop=True)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error loading historical trajectory data: {e}")

    return USER_HISTORY

def time_distribution_summary(user_data):
    """Generate a summary of user's time preferences."""
    if not pd.api.types.is_datetime64_any_dtype(user_data['UTCTimeOffset']):
        user_data['UTCTimeOffset'] = pd.to_datetime(user_data['UTCTimeOffset'])

    # Extract hour information
    user_data['hour'] = user_data['UTCTimeOffset'].dt.hour

    # Count frequency of visits for each hour
    hour_counts = user_data['hour'].value_counts().sort_index()
    hour_counts = hour_counts.sort_values(ascending=False)
    
    # Prepare explanation information
    time_explanation = [
        f"Time: {hour:02d}:00-{(hour+1)%24:02d}:00, Frequency: {count}"
        for hour, count in hour_counts.items()
    ]

    return time_explanation

def day_distribution_summary(user_data):
    """Generate a summary of user's day of week preferences."""
    user_data['day_of_week'] = user_data['UTCTimeOffset'].dt.dayofweek + 1  # Convert to 1-7 range

    # Count frequency of visits for each day of week
    day_counts = user_data['day_of_week'].value_counts().sort_index()
    
    # Map day numbers to day names
    day_names = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday"
    }
    
    # Prepare explanation information
    day_explanation = [
        f"Day: {day_names[day]}, Frequency: {count}"
        for day, count in day_counts.items()
    ]

    return day_explanation

def category_distribution_summary(user_data):
    """Generate a summary of user's POI category preferences."""
    # Count frequency of visits for each category
    category_counts = user_data['PoiCategoryName'].value_counts()
    
    # Prepare explanation information
    category_explanation = [
        f"Category: {category}, Frequency: {count}"
        for category, count in category_counts.items()
    ]

    return category_explanation

def poi_distribution_summary(user_data):
    """Generate a summary of user's POI preferences."""
    # Count frequency of visits for each POI
    poi_counts = user_data['PoiId'].astype(int).value_counts()
    
    # Prepare explanation information
    poi_explanation = [
        f"POI ID: {poi_id}, Frequency: {count}"
        for poi_id, count in poi_counts.items()
    ]

    return poi_explanation

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the geodesic distance between two points (lat1, lon1) and (lat2, lon2).
    """
    return geodesic((lat1, lon1), (lat2, lon2)).meters  # Distance in meters

def calculate_poi_transitions(user_data):
    """Calculate POI-to-POI transitions for a single user."""
    poi_transitions = defaultdict(int)
    
    # Loop through the user's trajectory data and count POI transitions
    for i in range(1, len(user_data)):
        prev_poi = user_data.iloc[i-1]['PoiId']
        curr_poi = user_data.iloc[i]['PoiId']
        poi_transitions[(prev_poi, curr_poi)] += 1
    
    return poi_transitions

def calculate_category_transitions(user_data):
    """Calculate Category-to-Category transitions for a single user."""
    category_transitions = defaultdict(int)
    
    # Loop through the user's trajectory data and count category transitions
    for i in range(1, len(user_data)):
        prev_category = user_data.iloc[i-1]['PoiCategoryName']
        curr_category = user_data.iloc[i]['PoiCategoryName']
        category_transitions[(prev_category, curr_category)] += 1
    
    return category_transitions

def global_transition_statistics(USER_HISTORY):
    """Calculate global POI and category transition frequencies for all users."""
    # Initialize dictionaries to store global transitions
    global_poi_transitions = defaultdict(int)
    global_category_transitions = defaultdict(int)
    
    # Iterate over all users and calculate transitions
    for user_id, user_data in USER_HISTORY.items():
        # Calculate transitions for each user
        poi_transitions = calculate_poi_transitions(user_data)
        category_transitions = calculate_category_transitions(user_data)
        
        # Aggregate global transitions
        for transition, count in poi_transitions.items():
            global_poi_transitions[transition] += count
        for transition, count in category_transitions.items():
            global_category_transitions[transition] += count
    
    return global_poi_transitions, global_category_transitions

def parse_user_data(descriptions):
    """Parse the input descriptions and extract the necessary data."""
    data = []
    
    # Define a regular expression to capture the time, POI ID, and Category
    pattern = r"At time (\S+ \S+), the user arrives at POI (\d+) \(Category: ([^\)]+)\)"
    # Iterate over each description
    for desc in descriptions:
        # print(desc)
        match = re.search(pattern, desc)
        if match:
            time = match.group(1)  # Extracted time
            poi_id = int(match.group(2))  # Extracted POI ID
            category = match.group(3)  # Extracted Category
            data.append({'UTCTimeOffset': time, 'PoiId': poi_id, 'PoiCategoryName': category})
            
    if not data:
        # If empty, return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['UTCTimeOffset', 'PoiId', 'PoiCategoryName'])

    # Convert the list of dictionaries into a pandas DataFrame
    user_data = pd.DataFrame(data)
    # Convert the time string to a datetime object
    user_data['UTCTimeOffset'] = pd.to_datetime(user_data['UTCTimeOffset'])
    # Remove duplicate rows based on all columns
    user_data = user_data.drop_duplicates()
    
    return user_data

def generate_trajectory_description(current_traj_data, hist_traj_data):
    """Generate a detailed trajectory description with time and distance between POIs."""
    current_trajectory_description = []
    history_trajectory_description = []
    if not hist_traj_data.empty:
        num_rows = hist_traj_data.shape[0]
        curr_row = hist_traj_data.iloc[0]
        curr_poi_id = curr_row['PoiId'].astype(int)
        curr_poi_category = curr_row['PoiCategoryName']
        curr_time = pd.to_datetime(curr_row['UTCTimeOffset'])
        # 确保有至少两行数据才能计算时间差和停留时间
        if num_rows == 1:
            history_trajectory_description.append(
                f"At time {curr_time.strftime('%Y-%m-%d %H:%M:%S')}, the user arrives at POI {curr_poi_id} "
                f"(Category: {curr_poi_category})"
            )
        else:
            next_row = hist_traj_data.iloc[1]
            # 计算停留时间（时间差）
            next_time = pd.to_datetime(next_row['UTCTimeOffset'])
            stay_duration = (next_time - curr_time).total_seconds() / 60  # 单位：分钟
            history_trajectory_description.append(
                    f"At time {curr_time.strftime('%Y-%m-%d %H:%M:%S')}, the user arrives at POI {curr_poi_id} "
                    f"(Category: {curr_poi_category}),"
                    f"and stays for {stay_duration:.2f} minutes."
                )
        # 计算每个相邻POI之间的时间差和距离差
        for i in range(1, len(hist_traj_data)):  # Fix: avoid out of index error for the last row
            prev_row = hist_traj_data.iloc[i-1]
            curr_row = hist_traj_data.iloc[i]
            if i == len(hist_traj_data)-1 :
                next_row = current_traj_data.iloc[0]
            else : next_row = hist_traj_data.iloc[i+1]
            
            # 获取当前POI和上一个POI的信息
            prev_poi_id = prev_row['PoiId'].astype(int)
            curr_poi_id = curr_row['PoiId'].astype(int)
            prev_poi_category = prev_row['PoiCategoryName']
            curr_poi_category = curr_row['PoiCategoryName']
            
            # 获取当前POI的经纬度
            prev_lat = prev_row['Latitude']
            prev_lon = prev_row['Longitude']
            curr_lat = curr_row['Latitude']
            curr_lon = curr_row['Longitude']
            
            # 计算停留时间（时间差）
            next_time = pd.to_datetime(next_row['UTCTimeOffset'])
            curr_time = pd.to_datetime(curr_row['UTCTimeOffset'])
            stay_duration = (next_time - curr_time).total_seconds() / 60  # 单位：分钟
            
            # 计算POI间的地理距离
            distance = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).meters  # 单位：米
            
            # 构建描述，时间格式改为包含完整日期和时间（YYYY-MM-DD HH:MM:SS）
            history_trajectory_description.append(
                f"At time {curr_time.strftime('%Y-%m-%d %H:%M:%S')}, the user arrives at POI {curr_poi_id} "
                f"(Category: {curr_poi_category}), which is {distance:.2f} meters away from POI {prev_poi_id}, "
                f"and stays for {stay_duration:.2f} minutes."
            )
    
    for i in range(0, len(current_traj_data)-1):  # Fix: avoid out of index error for the last row
              # 计算停留时间（时间差）
        curr_row = current_traj_data.iloc[i]    
        next_row = current_traj_data.iloc[i+1]
        next_time = pd.to_datetime(next_row['UTCTimeOffset'])
        curr_time = pd.to_datetime(curr_row['UTCTimeOffset'])
        stay_duration = (next_time - curr_time).total_seconds() / 60  # 单位：分钟
        curr_poi_id = curr_row['PoiId'].astype(int)
        curr_poi_category = curr_row['PoiCategoryName']
        if i==0 :     
            if hist_traj_data.empty: current_trajectory_description.append(
                f"At time {curr_time.strftime('%Y-%m-%d %H:%M:%S')}, the user arrives at POI {curr_poi_id} "
                f"(Category: {curr_poi_category}),"
                f"and stays for {stay_duration:.2f} minutes."
            )
            else: 
                prev_row = hist_traj_data.iloc[-1]
                prev_poi_id = prev_row['PoiId'].astype(int)
                prev_poi_category = prev_row['PoiCategoryName']
                
                # 获取当前POI的经纬度
                prev_lat = prev_row['Latitude']
                prev_lon = prev_row['Longitude']
                curr_lat = curr_row['Latitude']
                curr_lon = curr_row['Longitude']
                
                # 计算POI间的地理距离
                distance = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).meters  # 单位：米
                
                # 构建描述，时间格式改为包含完整日期和时间（YYYY-MM-DD HH:MM:SS）
                current_trajectory_description.append(
                    f"At time {curr_time.strftime('%Y-%m-%d %H:%M:%S')}, the user arrives at POI {curr_poi_id} "
                    f"(Category: {curr_poi_category}), which is {distance:.2f} meters away from POI {prev_poi_id}, "
                    f"and stays for {stay_duration:.2f} minutes."
                )
        else : 
            prev_row = current_traj_data.iloc[i-1]
            # 获取当前POI和上一个POI的信息
            prev_poi_id = prev_row['PoiId'].astype(int)
            prev_poi_category = prev_row['PoiCategoryName']
            
            # 获取当前POI的经纬度
            prev_lat = prev_row['Latitude']
            prev_lon = prev_row['Longitude']
            curr_lat = curr_row['Latitude']
            curr_lon = curr_row['Longitude']
            
            # 计算POI间的地理距离
            distance = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).meters  # 单位：米
            
            # 构建描述，时间格式改为包含完整日期和时间（YYYY-MM-DD HH:MM:SS）
            current_trajectory_description.append(
                f"At time {curr_time.strftime('%Y-%m-%d %H:%M:%S')}, the user arrives at POI {curr_poi_id} "
                f"(Category: {curr_poi_category}), which is {distance:.2f} meters away from POI {prev_poi_id}, "
                f"and stays for {stay_duration:.2f} minutes."
            )
    curr_row = current_traj_data.iloc[-1]
    curr_poi_id = curr_row['PoiId'].astype(int)
    curr_poi_category = curr_row['PoiCategoryName']
    curr_time = pd.to_datetime(curr_row['UTCTimeOffset'])

    if len(current_traj_data) > 1:
        prev_row = current_traj_data.iloc[-2]
        prev_poi_id = prev_row['PoiId'].astype(int)
   
        prev_lat = prev_row['Latitude']
        prev_lon = prev_row['Longitude']
        curr_lat = curr_row['Latitude']
        curr_lon = curr_row['Longitude']
        
        # 计算POI间的地理距离
        distance = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).meters  # 单位：米
            
        current_trajectory_description.append(
                f"At time {curr_time.strftime('%Y-%m-%d %H:%M:%S')}, the user arrives at POI {curr_poi_id} "
                f"(Category: {curr_poi_category}), which is {distance:.2f} meters away from POI {prev_poi_id}. "
            )
    else: 
        current_trajectory_description.append(
            f"At time {curr_time.strftime('%Y-%m-%d %H:%M:%S')}, the user arrives at POI {curr_poi_id} "
            f"(Category: {curr_poi_category})."
        )
        
        
    return history_trajectory_description, current_trajectory_description

def generate_trajectory_descriptions(main_data):
    """Generate current and historical trajectory descriptions for each user."""
    user_descriptions = {}
    # path = f'../datasets/{args.dataset_name}/preprocessed/'
    # main_data = pd.read_csv(f'{path}train_sample.csv')
    
    for user in main_data['UserId'].unique():
        user_data = main_data[main_data['UserId'] == user]
        user_data['UTCTimeOffsetEpoch'] = pd.to_datetime(user_data['UTCTimeOffsetEpoch'])
        all_trajectories = user_data.sort_values('UTCTimeOffsetEpoch')
        all_trajectories = all_trajectories.groupby('trajectory_id', as_index=False, sort=False)
        trajectory_list = [(traj_id, traj_data) for traj_id, traj_data in all_trajectories]

        # Initialize dictionary to store user descriptions
        user_descriptions[user] = {
            "current_trajectory_descriptions": [],  # List of lists to store all sub_trajectory descriptions
            "historical_trajectory_descriptions": []  # List of lists to store all sub_trajectory descriptions
        }

        index = min(0, len(trajectory_list)-1)
        for i in range(index, len(trajectory_list)):
            current_traj_id, current_traj_data = trajectory_list[i]
            
            # Get up to 15 most recent historical trajectories
            historical_trajectories = trajectory_list[max(0, i-15):i]  # Last 15 or all available
            
            # Generate current trajectory description for this sub_trajectory
            # current_trajectory_description = generate_trajectory_description(current_traj_data, hist_traj_data)
            # Save this sub_trajectory as a list (append it as a new list inside the outer list)
            # user_descriptions[user]["current_trajectory_descriptions"].append(current_trajectory_description)

            # Generate historical trajectory descriptions for this sub_trajectory
            historical_descriptions = []
            historical_data_list = []
            for hist_traj_id, hist_traj_data in historical_trajectories:
                historical_data_list.append(hist_traj_data)  # Add this trajectory's data
            if historical_data_list:
                # Combine historical data into one DataFrame if it exists
                historical_trajectory_data = pd.concat(historical_data_list)
                historical_description, current_trajectory_description = generate_trajectory_description(current_traj_data, historical_trajectory_data)
                # Save the descriptions
                user_descriptions[user]["current_trajectory_descriptions"].append(current_trajectory_description)
                user_descriptions[user]["historical_trajectory_descriptions"].append(historical_description)
            else:
                # If no historical data exists, save empty description for historical trajectory
                historical_trajectory_data = pd.DataFrame(historical_data_list)
                historical_description, current_trajectory_description = generate_trajectory_description(current_traj_data, historical_trajectory_data)
                user_descriptions[user]["current_trajectory_descriptions"].append(current_trajectory_description)
                user_descriptions[user]["historical_trajectory_descriptions"] = []
    return user_descriptions

def get_all_information_tool(user_id, data, Test=False):
    """
    Tool to get all information about a user's historical trajectory, including global transition statistics.
    
    Args:
        user_id: User ID (int or str)
        data: Dataset name (str)
        
    Returns:
        dict: A dictionary containing the user's historical data, summary statistics, and global transition frequencies.
    """
    try:
        # Load user history
        if Test:
            history_path = f'../datasets/{data}/preprocessed/test_sample_with_traj.csv'
        else: history_path = f'../datasets/{data}/preprocessed/train_sample.csv'
        USER_HISTORY = load_user_history(history_path)
        
        # Check if user exists in the dataset
        if int(user_id) not in USER_HISTORY:
            return {
                "error": f"User ID {user_id} not found in dataset {data}."
            }
        
        # Get user data from history
        user_data = USER_HISTORY[int(user_id)]
        
        # Generate summaries based on user data
        time_summary = time_distribution_summary(user_data)
        day_summary = day_distribution_summary(user_data)
        category_summary = category_distribution_summary(user_data)
        poi_summary = poi_distribution_summary(user_data)
        
        # Calculate global transition statistics
        global_poi_transitions, global_category_transitions = global_transition_statistics(USER_HISTORY)
        
        # Prepare the response
        response = {
            "user_id": user_id,
            "dataset": data,
            "time_distribution": time_summary,
            "day_distribution": day_summary,
            "category_distribution": category_summary,
            "poi_distribution": poi_summary,
            "trajectory_length": len(user_data),
            "trajectory_data": user_data.to_dict(orient='records'),
            "global_poi_transitions": dict(global_poi_transitions),
            "global_category_transitions": dict(global_category_transitions)
        }
        
        return response
    
    except FileNotFoundError:
        return {
            "error": f"Dataset file for {data} not found at path {history_path}."
        }
    
    except Exception as e:
        return {
            "error": f"An error occurred: {str(e)}"
        }
        
def get_all_information_tool_data(user_id, user_data):
    """
    Tool to get all information about a user's historical trajectory, including global transition statistics.
    
    Args:
        user_id: User ID (int or str)
        data: Dataset name (str)
        
    Returns:
        dict: A dictionary containing the user's historical data, summary statistics, and global transition frequencies.
    """
    try:
        # Load user history
        # if Test:
        #     history_path = f'../datasets/{data}/preprocessed/test_sample_with_traj.csv'
        # else: history_path = f'../datasets/{data}/preprocessed/train_sample.csv'
        # USER_HISTORY = load_user_history(history_path)
        
        # # Check if user exists in the dataset
        # if int(user_id) not in USER_HISTORY:
        #     return {
        #         "error": f"User ID {user_id} not found in dataset {data}."
        #     }
        
        # # Get user data from history
        # user_data = USER_HISTORY[int(user_id)]
        
        data = 'nyc'
        # Generate summaries based on user data
        time_summary = time_distribution_summary(user_data)
        day_summary = day_distribution_summary(user_data)
        category_summary = category_distribution_summary(user_data)
        poi_summary = poi_distribution_summary(user_data)
        
        # Calculate global transition statistics
        # global_poi_transitions, global_category_transitions = global_transition_statistics(USER_HISTORY)
        
        # Prepare the response
        response = {
            "user_id": user_id,
            "dataset": data,
            "time_distribution": time_summary,
            "day_distribution": day_summary,
            "category_distribution": category_summary,
            "poi_distribution": poi_summary,
            "trajectory_length": len(user_data),
            "trajectory_data": user_data.to_dict(orient='records'),
            # "global_poi_transitions": dict(global_poi_transitions),
            # "global_category_transitions": dict(global_category_transitions)
        }
        
        return response
    
    except FileNotFoundError:
        return {
            "error": f"Dataset file for {user_data} not found."
        }
    
    except Exception as e:
        return {
            "error": f"An error occurred: {str(e)}"
        }
        
def process_global_transitions(response):
    """
    Process and output the top 20 POI transitions and top 10 category transitions.
    
    Args:
        response (dict): The response dictionary from get_all_information_tool.
        
    Returns:
        None: Prints the results directly.
    """
    # Retrieve global transitions
    global_poi_transitions = response.get("global_poi_transitions", {})
    global_category_transitions = response.get("global_category_transitions", {})

    # Sort the POI transitions by frequency in descending order and get top 20
    sorted_poi_transitions = sorted(global_poi_transitions.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Sort the Category transitions by frequency in descending order and get top 10
    sorted_category_transitions = sorted(global_category_transitions.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Output top 20 POI transitions
    print("\nTop 20 POI Transitions:")
    for (prev_poi, curr_poi), count in sorted_poi_transitions:
        print(f"POI {prev_poi} -> POI {curr_poi}: {count} times")
    
    # Output top 10 Category transitions
    print("\nTop 10 Category Transitions:")
    for (prev_category, curr_category), count in sorted_category_transitions:
        print(f"Category {prev_category} -> Category {curr_category}: {count} times")


# # Example usage:
# user_id = 688  # Example user ID
# data = 'nyc'  # Example dataset name

# # # Call get_all_information_tool to get the response
# # response = get_all_information_tool(user_id, data)

# # # After getting the response, process and output POI and category transitions
# # process_global_transitions(response)

# path = f'../datasets/nyc/preprocessed/'
# user_descriptions = generate_trajectory_descriptions(path)
# print(2)
# # 获取指定用户的轨迹描述
# user_trajectory = user_descriptions.get(user_id, None)

# # 如果用户存在，打印当前轨迹和历史轨迹
# if user_trajectory:
#     print(f"User {user_id} Current Trajectory Description:\n")
#     print(user_trajectory["current_trajectory_descriptions"])
    
#     print(f"\nUser {user_id} Historical Trajectory Descriptions:\n")
#     for hist_desc in user_trajectory["historical_trajectory_descriptions"]:
#         print(hist_desc)
# else:
#     print(f"User {user_id} not found in the dataset.")