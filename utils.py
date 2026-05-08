import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_all_user_history(history_path):
    """Load historical trajectory data for all users."""
    all_data = pd.read_csv(history_path, parse_dates=['UTCTimeOffset'])
    all_data['day_of_week'] = all_data['UTCTimeOffset'].dt.dayofweek + 1  # Adjusted to match 1 = Monday, 7 = Sunday
    return all_data

def add_time_segment(all_data):
    """Assign each check-in to a specific time segment."""
    bins = [0, 6, 12, 18, 24]  # [00:00-06:00, 06:00-12:00, 12:00-18:00, 18:00-24:00]
    labels = ['00:00-06:00', '06:00-12:00', '12:00-18:00', '18:00-24:00']
    all_data['time_segment'] = pd.cut(all_data['UTCTimeOffset'].dt.hour, bins=bins, labels=labels, right=False)
    return all_data

def poi_frequency_summary(all_data):
    """Generate a summary of POI frequency for all users."""
    poi_counts = all_data['PoiId'].value_counts()
    poi_explanation = [
        f"POI ID: {poi_id}, Frequency: {count}"
        for poi_id, count in poi_counts.items()
    ]
    return poi_explanation

def category_frequency_summary(all_data):
    """Generate a summary of POI category frequency for all users."""
    category_counts = all_data['PoiCategoryName'].value_counts()
    category_explanation = [
        f"Category: {category}, Frequency: {count}"
        for category, count in category_counts.items()
    ]
    return category_explanation

def time_frequency_summary(all_data):
    """Generate a summary of time-based visit frequency for all users."""
    all_data['hour'] = all_data['UTCTimeOffset'].dt.hour
    hour_counts = all_data['hour'].value_counts().sort_index()
    time_explanation = [
        f"Time: {hour:02d}:00-{(hour+1)%24:02d}:00, Frequency: {count}"
        for hour, count in hour_counts.items()
    ]
    return time_explanation

def day_frequency_summary(all_data):
    """Generate a summary of day-based visit frequency for all users."""
    day_counts = all_data['day_of_week'].value_counts().sort_index()
    day_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
    day_explanation = [
        f"Day: {day_names[day]}, Frequency: {count}"
        for day, count in day_counts.items()
    ]
    return day_explanation

def transition_probability_summary(all_data):
    """Calculate the transition probabilities between POIs or categories."""
    # Add time segment column
    all_data = add_time_segment(all_data)
    
    # Compute transition probabilities between POIs
    all_data['next_poi'] = all_data['PoiId'].shift(-1)
    poi_transitions = all_data.groupby(['PoiId', 'next_poi']).size().reset_index(name='transition_count')
    poi_transitions['transition_prob'] = poi_transitions['transition_count'] / poi_transitions.groupby('PoiId')['transition_count'].transform('sum')

    # Transition explanation for POI
    poi_transition_explanation = [
        f"POI {row['PoiId']} -> POI {row['next_poi']}, Transition Probability: {row['transition_prob']:.4f}"
        for _, row in poi_transitions.iterrows()
    ]

    # Compute transition probabilities between categories
    all_data['next_category'] = all_data['PoiCategoryName'].shift(-1)
    category_transitions = all_data.groupby(['PoiCategoryName', 'next_category']).size().reset_index(name='transition_count')
    category_transitions['transition_prob'] = category_transitions['transition_count'] / category_transitions.groupby('PoiCategoryName')['transition_count'].transform('sum')

    # Transition explanation for categories
    category_transition_explanation = [
        f"Category {row['PoiCategoryName']} -> Category {row['next_category']}, Transition Probability: {row['transition_prob']:.4f}"
        for _, row in category_transitions.iterrows()
    ]
    
    # Compute transition probabilities between time segments
    all_data['next_time_segment'] = all_data['time_segment'].shift(-1)
    time_transitions = all_data.groupby(['time_segment', 'next_time_segment']).size().reset_index(name='transition_count')
    time_transitions['transition_prob'] = time_transitions['transition_count'] / time_transitions.groupby('time_segment')['transition_count'].transform('sum')

    # Transition explanation for time segments
    time_transition_explanation = [
        f"Time {row['time_segment']} -> Time {row['next_time_segment']}, Transition Probability: {row['transition_prob']:.4f}"
        for _, row in time_transitions.iterrows()
    ]

    return poi_transition_explanation, category_transition_explanation, time_transition_explanation


def plot_poi_frequency_distribution(all_data, max_frequency=50, save_path=None):
    """Plot the distribution of POI visit frequencies (from frequency 1 to max_frequency) as a line plot."""
    
    # 1. Calculate POI visit frequencies
    poi_counts = all_data['PoiId'].value_counts()
    
    # 2. Count how many POIs have the same visit frequency (e.g., 1 visit, 2 visits, etc.)
    frequency_distribution = poi_counts.value_counts().sort_index(ascending=True)

    # 3. Only consider frequencies from 1 to max_frequency
    frequency_distribution = frequency_distribution[1:max_frequency+1]

    # 4. Set up the figure
    plt.figure(figsize=(14, 6))

    # 5. Plot the line chart
    sns.lineplot(x=frequency_distribution.index, y=frequency_distribution.values, marker='o', color='skyblue')

    # 6. Add labels and title
    plt.xlabel('Visit Frequency', fontsize=12)
    plt.ylabel('Number of POIs', fontsize=12)
    plt.title(f'POI Visit Frequency Distribution (Line Plot, 1 to {max_frequency})', fontsize=14)

    # 7. Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha='right', fontsize=10)

    # 8. Optionally, save the plot
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)  # Save the plot as a file
        print(f"Plot saved as {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def plot_category_frequency_distribution(all_data, save_path=None):
    """Plot the distribution of top 100 POI category visit frequencies with binned frequency intervals."""
    
    # 1. Calculate category visit frequencies
    category_counts = all_data['PoiCategoryId'].value_counts()

    # 2. Select the top 100 categories based on visit frequency
    top_100_categories = category_counts.head(100)

    # 3. Create frequency bins (intervals of 5)
    bins = range(0, top_100_categories.max() + 5, 5)
    binned_counts = pd.cut(top_100_categories, bins=bins, right=False).value_counts().sort_index()

    # 4. Set up the figure
    plt.figure(figsize=(14, 6))

    # 5. Plot the bar chart for binned frequency distribution
    sns.barplot(x=binned_counts.index.astype(str), y=binned_counts.values, color='skyblue')

    # 6. Add labels and title
    plt.xlabel('Visit Frequency (Grouped in Bins)', fontsize=12)
    plt.ylabel('Number of Categories', fontsize=12)
    plt.title('Top 100 Category Visit Frequency Distribution (Binned)', fontsize=14)

    # 7. Rotate x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=10)

    # 8. Optionally, save the plot
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)  # Save the plot as a file
        print(f"Plot saved as {save_path}")
    else:
        plt.tight_layout()
        plt.show()

def get_all_global_information(history_path):
    """Tool to get all global information about all users' historical trajectories."""
    try:
        # Load all users' data
        all_data = load_all_user_history(history_path)

        # Generate global summaries
        poi_summary = poi_frequency_summary(all_data)
        category_summary = category_frequency_summary(all_data)
        time_summary = time_frequency_summary(all_data)
        day_summary = day_frequency_summary(all_data)

        poi_transition_summary, category_transition_summary, time_transition_summary = transition_probability_summary(all_data)

        
        # Prepare response
        response = {
            "poi_distribution": poi_summary,
            "category_distribution": category_summary,
            "time_distribution": time_summary,
            "day_distribution": day_summary,
            "poi_transition_probabilities": poi_transition_summary,
            "category_transition_probabilities": category_transition_summary,
            "time_transition_probabilities": time_transition_summary,
            "total_data_length": len(all_data),
            "total_users": len(all_data['UserId'].unique())
        }

        return response
    
    except Exception as e:
        return f"Error getting global information: {e}"

def check_missing_poi_ids(df, poi_column='PoiId', valid_range=(0, 4981)):
    """
    检查 POI ID 在指定范围内是否有缺失的 POI ID。
    
    Args:
    - df: 包含 POI ID 的数据框。
    - poi_column: 数据框中存储 POI ID 的列名。
    - valid_range: 一个包含有效 POI ID 范围的元组 (min, max)。
    
    Returns:
    - missing_pois: 一个列表，包含缺失的 POI ID。
    """
    # 获取有效的 POI ID 范围
    valid_pois = set(range(valid_range[0], valid_range[1] + 1))
    
    # 获取数据中出现的 POI ID
    poi_in_data = set(df[poi_column].unique())
    
    # 查找缺失的 POI ID
    missing_pois = valid_pois - poi_in_data

    if len(missing_pois) > 0:
        print(f"Missing POI IDs: {sorted(list(missing_pois))}")
    else:
        print("No POI IDs are missing in the specified range.")

    return sorted(list(missing_pois))

# Example usage
history_path = 'datasets/nyc/preprocessed/train_sample.csv'
# response = get_all_global_information(history_path)
all_data = load_all_user_history(history_path)
# plot_poi_frequency_distribution(all_data, 100, save_path='poi_frequency_distribution_hot.png')
# plot_category_frequency_distribution(all_data, save_path='category_frequency_distribution.png')

# Print out the generated transition probabilities
# for entry in response['poi_distribution']:
#     print(entry)
# for entry in response['category_distribution']:
#     print(entry)
# for entry in response['time_transition_probabilities']:
#     print(entry)


# 检查缺失的 POI ID
missing_pois = check_missing_poi_ids(all_data)

# 输出缺失的 POI ID
if missing_pois:
    print(f"Total {len(missing_pois)} missing POIs.")
else:
    print("No POI IDs are missing.")