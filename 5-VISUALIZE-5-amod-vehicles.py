import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import os
from collections import defaultdict
from typing import Dict, List, Union, Tuple
import multiprocessing
import pickle

# ================= 配置区域 (Global Config) =================

# Train Headway (min)
train_headway_list = [10, 20, 30]

# MoD Fleet Size
mod_fleet_size_list = [30, 50, 70, 90, 110, 130, 150]

# MaaS Platform Communication Strategy
maas_communication_strategy_list = ['default', 'TPCS']

# Random Seed
random_seed_list = [3, 6, 9]

# MoD Waiting Time Threshold (s)
mod_waiting_time_threshold_list = [300, 600, 900]
# MoD Detour Time Threshold (%)
mod_detour_time_threshold_list = [30, 60, 90]

# Demand Size
demand_size_list = [i for i in range(100, 1001, 100)]
# Demand Split Ratio (Intra Modal, %)
demand_split_ratio_list = [0, 20, 40, 60, 80]

# Total Simulation Time (s)
total_sim_time = [0, 10800]  # 3 hours
# Warm-up Time (s)
warmup_time = 3600  # 1 hour
# Simulation Time Period (s)
time_period = [warmup_time, total_sim_time[1] + warmup_time]  # 1h + 3h

amod_simulation_results_folder = r"D:\projects\fleetpy\github\ptbroker\studies\j26-tpcs\results"
save_folder = r'data\amod-sim-results'

# ================= 功能函数 (Functions) =================

def get_boarding_vehicles_at_pos(
    file_path: str, 
    target_pos_id: Union[str, int], 
    target_operator_id: Union[str, int]
) -> Dict[int, List[str]]:
    
    boarding_schedule = defaultdict(list)
    
    target_pos_id = str(target_pos_id)
    target_operator_id = str(target_operator_id)

    if not os.path.exists(file_path):
        # 仅在非多进程调试时建议开启，多进程下print可能会混乱，或者使用logging
        # print(f"Cannot find file: {file_path}")
        return {}

    try:
        # 优化：只读取需要的列，减少内存消耗和解析时间
        use_cols = ['operator_id', 'status', 'start_pos', 'vehicle_id', 'start_time', 'end_time']
        df = pd.read_csv(file_path, usecols=use_cols)
        
        df['operator_id'] = df['operator_id'].astype(str)
        df['status'] = df['status'].astype(str)
        df['start_pos'] = df['start_pos'].astype(str)

        mask = (df['operator_id'] == target_operator_id) & (df['status'] == 'boarding')
        filtered_df = df[mask].copy()

        if filtered_df.empty:
            return {}

        # Delete '-1;-1'
        filtered_df['pos_id_parsed'] = filtered_df['start_pos'].str.split(';').str[0]
        
        # Filter by position ID
        target_df = filtered_df[filtered_df['pos_id_parsed'] == target_pos_id]

        # Iterate over the filtered data and expand the time intervals
        for _, row in target_df.iterrows():
            try:
                veh_id = str(row['vehicle_id'])
                start_time = int(float(row['start_time']))
                end_time = int(float(row['end_time']))
                
                for t in range(start_time, end_time + 1):
                    boarding_schedule[t].append(veh_id)
                    
            except (ValueError, TypeError):
                continue

    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return {}

    return dict(boarding_schedule)

def process_scenario(scenario_combination: Tuple) -> List[Tuple]:
    """
    处理单个场景的 Worker 函数。
    返回一个包含 (key, value) 元组的列表，以便主进程更新字典。
    """
    (
        random_seed,
        fleet_size,
        demand_size,
        demand_split_ratio,
        broker_type,
        op_max_detour_time_factor,
        op_max_wait_time,
        train_headway
    ) = scenario_combination

    scenario_name = f"amod-{demand_size}-{demand_split_ratio}-{fleet_size}-{broker_type}-{op_max_detour_time_factor}-{op_max_wait_time}-{train_headway}-{random_seed}-{time_period[0]}-{time_period[1]}"

    vehicle_trajectory_filepath = os.path.join(amod_simulation_results_folder, scenario_name, '2-0_op-stats.csv')

    # 执行读取操作
    left_mobility_hub_boarding_records = get_boarding_vehicles_at_pos(vehicle_trajectory_filepath, 120, 0)
    right_mobility_hub_boarding_records = get_boarding_vehicles_at_pos(vehicle_trajectory_filepath, 241, 0)

    # 构造结果数据的 Key 部分
    base_key_tuple = (
        random_seed,
        fleet_size,
        demand_size,
        demand_split_ratio,
        broker_type,
        op_max_detour_time_factor,
        op_max_wait_time,
        train_headway
    )

    # 返回两个结果（Left 和 Right）
    results = [
        (base_key_tuple + ('left',), left_mobility_hub_boarding_records),
        (base_key_tuple + ('right',), right_mobility_hub_boarding_records)
    ]
    
    return results

# ================= 主执行块 (Main Execution) =================

if __name__ == '__main__':
    # 1. 生成所有场景组合
    all_scenario_combinations = list(product(
        random_seed_list,
        mod_fleet_size_list,
        demand_size_list,
        demand_split_ratio_list,
        maas_communication_strategy_list,
        mod_detour_time_threshold_list,
        mod_waiting_time_threshold_list,
        train_headway_list
    ))
    
    print(f"Total scenarios to process: {len(all_scenario_combinations)}")

    # 2. 准备最终结果字典
    all_mobility_hub_boarding_records = {}

    # 3. 设置多进程 Pool
    # cpu_count() 可以获取核心数，您可以手动指定如 processes=8
    num_processes = min(multiprocessing.cpu_count(), len(all_scenario_combinations))
    print(f"Starting multiprocessing pool with {num_processes} workers...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 可以提高效率（处理完一个就返回一个），并配合 tqdm 显示进度
        # chunksize 设置为 1 比较适合文件 I/O 密集型任务
        results_iterator = pool.imap_unordered(process_scenario, all_scenario_combinations, chunksize=1)
        
        for batch_results in tqdm(results_iterator, total=len(all_scenario_combinations)):
            # process_scenario 返回的是一个列表，包含 left 和 right 的结果
            for key, data in batch_results:
                all_mobility_hub_boarding_records[key] = data

    # 4. 保存结果
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    save_path = os.path.join(save_folder, 'all_mobility_hub_boarding_records.pkl')
    print(f"Saving results to {save_path}...")

    with open(save_path, 'wb') as f:
        pickle.dump(all_mobility_hub_boarding_records, f)

    print("Done.")