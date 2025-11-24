import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from itertools import product
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入你的自定义模块
from src.pt.PTOperator import PTOperator

# ================= 全局变量占位符 =================
# 这些变量将在每个子进程内部被初始化
train_10_pt_control = None
train_20_pt_control = None
train_30_pt_control = None
walking_time_matrix = None
node_id_list = None

# ================= 子进程初始化函数 =================
def worker_init():
    """
    此函数会在每个子进程启动时运行一次。
    用于在子进程内存中安全地初始化 C++ / Cython 对象和加载只读数据。
    """
    global train_10_pt_control, train_20_pt_control, train_30_pt_control
    global walking_time_matrix, node_id_list
    
    # 1. 加载网络数据
    network_name = "11-500"
    network_path = f"data/network/{network_name}/"
    
    try:
        # 这里不需要加载所有 csv，只需要计算用到的 matrix
        # 即使父进程加载了，Windows 下子进程也无法直接访问，必须重载
        distance_matrix = np.load(network_path + "dist_matrix.npy")
        
        # Walking Speed (m/s)
        WALKING_SPEED = 1.33
        with np.errstate(divide='ignore', invalid='ignore'):
            walking_time_matrix = distance_matrix / WALKING_SPEED
            
        # 简单的节点列表加载
        node_df = pd.read_csv(network_path + "nodes.csv")
        node_id_list = node_df['node_index'].tolist()
        
    except FileNotFoundError:
        # 仅用于防止本地测试报错，实际运行应确保文件存在
        walking_time_matrix = np.zeros((1000, 1000))
        node_id_list = []

    # 2. 初始化 PTOperator (C++ 对象)
    # 在子进程内初始化是处理 C++ 扩展最安全的方式，避免了 Fork 导致的潜在状态损坏
    try:
        train_10_gtfs_dir = r"data/gtfs/train/train_headway_10/matched"
        train_10_pt_control = PTOperator(train_10_gtfs_dir, print_logs=False)

        train_20_gtfs_dir = r"data/gtfs/train/train_headway_20/matched"
        train_20_pt_control = PTOperator(train_20_gtfs_dir, print_logs=False)

        train_30_gtfs_dir = r"data/gtfs/train/train_headway_30/matched"
        train_30_pt_control = PTOperator(train_30_gtfs_dir, print_logs=False)
    except Exception as e:
        # 同样，仅为了防止无数据时崩溃
        pass

# ================= 核心处理函数 =================

def process_one_scenario(scenario_combination):
    """
    处理单个场景。注意：此函数内部直接使用全局变量（train_10_pt_control等），
    这些变量已经在 worker_init 中被初始化好了。
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

    STREET_STATION_TRANSFER_TIME = 60
    DEFAULT_BOARDING_TIME = 30
    
    # 这里的路径配置与你的环境一致
    demand_files_folder = "data/demand/11-500/amod"
    amod_request_level_analysis_folder = "data/amod-sim-results"
    amod_simulation_results_folder = "D:\\projects\\fleetpy\\github\\ptbroker\\studies\\j26-tpcs\\results"
    time_period = [3600, 10800 + 3600]

    demand_filepath = os.path.join(demand_files_folder, f"amod_ds{demand_size}_dsr{demand_split_ratio}_rs{random_seed}.csv")
    scenario_name = f"amod-{demand_size}-{demand_split_ratio}-{fleet_size}-{broker_type}-{op_max_detour_time_factor}-{op_max_wait_time}-{train_headway}-{random_seed}-{time_period[0]}-{time_period[1]}"
    amod_request_level_analysis_results_filepath = os.path.join(amod_request_level_analysis_folder, scenario_name, 'amod_request_level_analysis_results.csv')
    amod_simulation_results_scenario_folder = os.path.join(amod_simulation_results_folder, scenario_name)

    if not os.path.exists(demand_filepath) or not os.path.exists(amod_request_level_analysis_results_filepath):
        return f"Skipped: Files missing for {scenario_name}"

    try:
        demand = pd.read_csv(demand_filepath)
        amod_request_level_analysis_results = pd.read_csv(amod_request_level_analysis_results_filepath)
        user_stats = pd.read_csv(os.path.join(amod_simulation_results_scenario_folder, '1_user-stats.csv'))
    except Exception as e:
        return f"Error loading files for {scenario_name}: {e}"

    # 初始化新列
    amod_request_level_analysis_results = amod_request_level_analysis_results.assign(
        request_time=-1.0, fm_duration=-1.0, pt_duration=-1.0, 
        lm_duration=-1.0, pt_start_time=-1.0, lm_start_time=-1.0
    )

    # --- 优化后的 Inter-city 处理 ---
    intercity_indices = amod_request_level_analysis_results.index[
        amod_request_level_analysis_results['rq_type'] == 'inter'
    ].tolist()
    
    demand_time_map = dict(zip(demand['request_id'], demand['rq_time']))
    demand_start_map = dict(zip(demand['request_id'], demand['start']))
    demand_end_map = dict(zip(demand['request_id'], demand['end']))
    user_stats_grouped = user_stats.groupby('request_id')

    updates = {
        'request_time': {}, 'fm_duration': {}, 'pt_duration': {}, 
        'lm_duration': {}, 'pt_start_time': {}, 'lm_start_time': {}
    }

    # 引用全局变量
    global walking_time_matrix
    global train_10_pt_control, train_20_pt_control, train_30_pt_control

    for idx in intercity_indices:
        row = amod_request_level_analysis_results.iloc[idx]
        request_id = row['request_id']
        served_by_amod = row['served_by_amod']
        subnetwork = row['subnetwork']
        
        request_time = demand_time_map.get(request_id, -1)
        updates['request_time'][idx] = request_time

        fm_dur, pt_dur, lm_dur = -1.0, -1.0, -1.0
        pt_start, lm_start = -1.0, -1.0

        if served_by_amod:
            if request_id in user_stats_grouped.groups:
                stats_df = user_stats_grouped.get_group(request_id)
                fm_trip = stats_df[stats_df['sub_trip_id'] == 5]
                pt_trip = stats_df[stats_df['sub_trip_id'] == 6]
                lm_trip = stats_df[stats_df['sub_trip_id'] == 7]

                if not fm_trip.empty and not pt_trip.empty and not lm_trip.empty:
                    pt_start = pt_trip['earliest_pickup_time'].values[0]
                    lm_start = lm_trip['earliest_pickup_time'].values[0]
                    lm_end = lm_trip['dropoff_time'].values[0]

                    fm_dur = pt_start - request_time
                    pt_dur = lm_start - pt_start
                    lm_dur = lm_end - lm_start + DEFAULT_BOARDING_TIME
        else:
            origin_node = demand_start_map.get(request_id)
            destination_node = demand_end_map.get(request_id)

            if subnetwork == 'left':
                start_station_street_node = 120
                start_station_id = "MH-L"
                end_station_street_node = 241
                end_station_id = "MH-R"
            else:
                start_station_street_node = 241
                start_station_id = "MH-R"
                end_station_street_node = 120
                end_station_id = "MH-L"
            
            # 使用全局初始化的 walking_time_matrix
            if walking_time_matrix is not None:
                fm_dur = walking_time_matrix[origin_node, start_station_street_node] + STREET_STATION_TRANSFER_TIME
                lm_dur = walking_time_matrix[end_station_street_node, destination_node] + STREET_STATION_TRANSFER_TIME
            
            pt_start = request_time + fm_dur
            arrival_datetime = datetime(2024, 1, 1, 0, 0, 0) + pd.to_timedelta(pt_start, unit='s')

            # 使用全局初始化的 pt_control
            pt_control = None
            if train_headway == 10:
                pt_control = train_10_pt_control
            elif train_headway == 20:
                pt_control = train_20_pt_control
            else:
                pt_control = train_30_pt_control
            
            if pt_control:
                try:
                    # 调用 C++ 扩展方法
                    pt_dur = pt_control.return_fastest_pt_journey_1to1(
                        start_station_id, end_station_id, arrival_datetime, 3, detailed=False
                    )['duration']
                except Exception:
                    pt_dur = 0
            else:
                pt_dur = 0

            lm_start = pt_start + pt_dur

        updates['fm_duration'][idx] = fm_dur
        updates['pt_duration'][idx] = pt_dur
        updates['lm_duration'][idx] = lm_dur
        updates['pt_start_time'][idx] = pt_start
        updates['lm_start_time'][idx] = lm_start

    # 批量写入
    for col, data_dict in updates.items():
        if data_dict:
             amod_request_level_analysis_results.loc[list(data_dict.keys()), col] = list(data_dict.values())

    # --- 优化后的 Intra-city 处理 ---
    intra_mask = amod_request_level_analysis_results['rq_type'] == 'intra'
    amod_request_level_analysis_results.loc[intra_mask, 'request_time'] = \
        amod_request_level_analysis_results.loc[intra_mask, 'request_id'].map(demand_time_map)

    amod_request_level_analysis_results.to_csv(amod_request_level_analysis_results_filepath, index=False)
    
    return f"Done: {scenario_name}"

# ================= 主程序入口 =================

if __name__ == '__main__':
    
    # 场景参数配置...
    train_headway_list = [10, 20, 30]
    mod_fleet_size_list = [30, 50, 70, 90, 110, 130, 150]
    maas_communication_strategy_list = ['default', 'TPCS']
    random_seed_list = [3, 6, 9]
    mod_waiting_time_threshold_list = [300, 600, 900]
    mod_detour_time_threshold_list = [30, 60, 90]
    demand_size_list = [i for i in range(100, 1001, 100)]
    demand_split_ratio_list = [0, 20, 40, 60, 80]

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
    
    print(f"Total scenarios: {len(all_scenario_combinations)}")
    
    # 注意：如果 PTOperator 占用内存很大（例如 2GB），请减少 max_workers 的数量
    # 否则 8 个进程可能会占用 16GB 内存导致崩溃
    max_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    # max_workers = 4 # 如果内存不足，手动设置一个较小的值
    
    print(f"Starting pool with {max_workers} workers...")
    
    # 关键修改：使用 initializer=worker_init
    # 这确保了每个进程在接收任务前，先在本地执行 worker_init 加载 C++ 对象
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init) as executor:
        futures = {executor.submit(process_one_scenario, combo): combo for combo in all_scenario_combinations}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
            except Exception as e:
                print(f"Task failed: {e}")