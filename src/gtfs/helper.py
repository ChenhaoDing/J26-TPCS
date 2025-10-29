import pandas as pd
from datetime import datetime, timedelta

def expand_timetable(base_trip_data, study_start_str, study_end_str, headway_minutes):
    """
    扩展单个基准行程的时刻表。

    参数:
    - base_trip_data (dict): 包含基准行程信息的字典。
    - study_start_str (str): 研究范围的开始时间 (HH:MM:SS)。
    - study_end_str (str): 研究范围的结束时间 (HH:MM:SS)。
    - headway_minutes (int): 发车间隔（分钟）。

    返回:
    - pd.DataFrame: 包含所有扩展行程的 DataFrame。
    """
    # 将时间字符串转换为 datetime 对象以便计算
    study_start_time = datetime.strptime(study_start_str, '%H:%M:%S')
    study_end_time = datetime.strptime(study_end_str, '%H:%M:%S')
    headway = timedelta(minutes=headway_minutes)

    # 将基准行程字典转换为 DataFrame
    base_df = pd.DataFrame(base_trip_data)
    
    # 将 DataFrame 中的时间字符串转换为 datetime 对象
    base_df['arrival_time'] = pd.to_datetime(base_df['arrival_time'], format='%H:%M:%S')
    base_df['departure_time'] = pd.to_datetime(base_df['departure_time'], format='%H:%M:%S')

    all_trips = [base_df]
    base_trip_id = base_df['trip_id'].iloc[0]

    # --- 向前扩展行程 ---
    current_trip_df = base_df.copy()
    counter = 1
    while True:
        next_trip_df = current_trip_df.copy()
        # 增加 headway 时间
        next_trip_df['arrival_time'] += headway
        next_trip_df['departure_time'] += headway
        
        # 检查新行程的开始时间是否在研究范围内
        if next_trip_df['departure_time'].iloc[0] >= study_end_time:
            break
            
        # 更新 trip_id
        next_trip_df['trip_id'] = f"{base_trip_id}-{counter}"
        all_trips.append(next_trip_df)
        
        current_trip_df = next_trip_df
        counter += 1

    # --- 向后扩展行程 ---
    current_trip_df = base_df.copy()
    while True:
        prev_trip_df = current_trip_df.copy()
        # 减去 headway 时间
        prev_trip_df['arrival_time'] -= headway
        prev_trip_df['departure_time'] -= headway

        # 检查新行程的开始时间是否在研究范围内
        if prev_trip_df['departure_time'].iloc[0] < study_start_time:
            break

        # 更新 trip_id
        prev_trip_df['trip_id'] = f"{base_trip_id}-{counter}"
        # 将向后生成的行程插入到列表的开头，以保持时间顺序
        all_trips.insert(0, prev_trip_df)
        
        current_trip_df = prev_trip_df
        counter += 1
        
    # 合并这个基准线路产生的所有行程
    expanded_df = pd.concat(all_trips, ignore_index=True)

    # Add 0 to trip_id for base trip
    expanded_df.loc[expanded_df['trip_id'] == base_trip_id, 'trip_id'] = f"{base_trip_id}-0"
    return expanded_df