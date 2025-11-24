import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from itertools import product
from datetime import datetime
import time
from multiprocessing import Pool, cpu_count
import sys

# 假设 PTOperator 在 src/pt/PTOperator.py 中定义
# 确保此脚本可以导入 src 模块
try:
    from src.pt.PTOperator import PTOperator
except ImportError:
    print("错误：无法导入 'src.pt.PTOperator'。")
    print("请确保脚本从可以访问 'src' 目录的父目录运行。")
    sys.exit(1)


# --- Worker 初始化函数 ---
# 此函数在每个工作进程启动时运行一次
def init_worker(walking_matrix, pt_10, pt_20, pt_30, 
                demand_folder, analysis_folder, 
                transfer_time, sim_time_period):
    """
    初始化工作进程，将大型只读对象加载到
    每个工作进程的全局变量中。
    """
    # 将这些只读对象存储为此工作进程内的全局变量
    # 这避免了为每个任务序列化（pickle）它们的开销
    global g_walking_time_matrix, g_pt_10, g_pt_20, g_pt_30
    global g_demand_folder, g_analysis_folder, g_transfer_time, g_time_period
    
    g_walking_time_matrix = walking_matrix
    g_pt_10 = pt_10
    g_pt_20 = pt_20
    g_pt_30 = pt_30
    g_demand_folder = demand_folder
    g_analysis_folder = analysis_folder
    g_transfer_time = transfer_time
    g_time_period = sim_time_period

# --- Worker 函数 ---
# 此函数处理 *单个* 场景组合。
# 它必须是一个顶级函数才能被 multiprocessing 挑选。
def process_scenario(random_seed, fleet_size, demand_size, demand_split_ratio,
                     broker_type, op_max_detour_time_factor, op_max_wait_time,
                     train_headway):
    """
    处理单个场景组合的函数。
    它从工作进程的全局范围读取大型数据对象。
    """
    
    # 访问在 init_worker 中设置的全局变量
    # (读取全局变量不需要 'global' 关键字)
    
    # 在此函数内部导入所需的库
    # 这确保它们在工作进程中可用
    import pandas as pd
    import numpy as np
    import os
    from datetime import datetime

    try:
        # --- 1. 设置路径和名称 ---
        demand_filepath = os.path.join(g_demand_folder, f"amod_ds{demand_size}_dsr{demand_split_ratio}_rs{random_seed}.csv")

        scenario_name = (f"amod-{demand_size}-{demand_split_ratio}-{fleet_size}-{broker_type}-"
                         f"{op_max_detour_time_factor}-{op_max_wait_time}-{train_headway}-"
                         f"{random_seed}-{g_time_period[0]}-{g_time_period[1]}")

        amod_request_level_analysis_results_filepath = os.path.join(g_analysis_folder, scenario_name, 'amod_request_level_analysis_results.csv')

        # --- 2. 检查文件是否存在 ---
        if not os.path.exists(demand_filepath):
            return (scenario_name, "SKIPPED", f"缺少需求文件: {demand_filepath}")
        if not os.path.exists(amod_request_level_analysis_results_filepath):
             return (scenario_name, "SKIPPED", f"缺少结果文件: {amod_request_level_analysis_results_filepath}")

        # --- 3. 加载文件 ---
        demand = pd.read_csv(demand_filepath)
        amod_request_level_analysis_results = pd.read_csv(amod_request_level_analysis_results_filepath)

        # --- 4. 处理数据 (与原始循环主体相同) ---
        amod_request_level_analysis_results['served_by_walking'] = 0
        unserved_requests = amod_request_level_analysis_results[amod_request_level_analysis_results['served_by_amod'] == False]
        unserved_intra_requests = unserved_requests[unserved_requests['rq_type'] == 'intra']
        unserved_inter_requests = unserved_requests[unserved_requests['rq_type'] == 'inter']

        # 处理未服务的 intra 请求
        for idx, unserved_request in unserved_intra_requests.iterrows():
            request_id = unserved_request['request_id']
            origin_node = demand.loc[demand['request_id'] == request_id, 'start'].values[0]
            destination_node = demand.loc[demand['request_id'] == request_id, 'end'].values[0]
            walking_time = g_walking_time_matrix[origin_node, destination_node]
            
            amod_request_level_analysis_results.loc[amod_request_level_analysis_results['request_id'] == request_id, 'served_by_walking'] = 1
            amod_request_level_analysis_results.loc[amod_request_level_analysis_results['request_id'] == request_id, 'total_journey_time'] = walking_time
        
        # 处理未服务的 inter 请求
        for idx, unserved_request in unserved_inter_requests.iterrows():
            request_id = unserved_request['request_id']
            origin_node = demand.loc[demand['request_id'] == request_id, 'start'].values[0]
            destination_node = demand.loc[demand['request_id'] == request_id, 'end'].values[0]
            subnetwork = unserved_request['subnetwork']
            request_time = demand.loc[demand['request_id'] == request_id, 'rq_time'].values[0]

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
            
            walking_time_to_station = g_walking_time_matrix[origin_node, start_station_street_node] + g_transfer_time
            walking_time_from_station = g_walking_time_matrix[end_station_street_node, destination_node] + g_transfer_time

            arrival_time_at_station = request_time + walking_time_to_station
            arrival_datetime = datetime(2024, 1, 1, 0, 0, 0) + pd.to_timedelta(arrival_time_at_station, unit='s')

            if train_headway == 10:
                pt_control = g_pt_10
            elif train_headway == 20:
                pt_control = g_pt_20
            else:
                pt_control = g_pt_30

            duration = pt_control.return_fastest_pt_journey_1to1(start_station_id, end_station_id, arrival_datetime, 3, detailed=False)['duration']
            total_journey_time = walking_time_to_station + duration + walking_time_from_station
            
            amod_request_level_analysis_results.loc[amod_request_level_analysis_results['request_id'] == request_id, 'served_by_walking'] = 1
            amod_request_level_analysis_results.loc[amod_request_level_analysis_results['request_id'] == request_id, 'total_journey_time'] = total_journey_time

        # --- 5. 保存更新的结果 ---
        amod_request_level_analysis_results.to_csv(amod_request_level_analysis_results_filepath, index=False)
        
        # 返回成功状态
        return (scenario_name, "SUCCESS", "Processed successfully")
        
    except Exception as e:
        # 在出错时返回错误信息
        return (scenario_name, "ERROR", str(e))

# --- 主执行函数 ---
def main():
    # --- 1. 加载所有共享数据 (在主进程中) ---
    print("正在加载网络文件...")
    network_name = "11-500"
    network_path = f"data/network/{network_name}/"
    distance_matrix = np.load(network_path + "dist_matrix.npy")

    WALKING_SPEED = 1.33
    STREET_STATION_TRANSFER_TIME = 60
    walking_time_matrix = distance_matrix / WALKING_SPEED
    print("网络文件加载完毕。")

    print("正在加载 PTOperator 数据...")
    train_10_gtfs_dir = r"data/gtfs/train/train_headway_10/matched"
    train_10_pt_control = PTOperator(train_10_gtfs_dir)

    train_20_gtfs_dir = r"data/gtfs/train/train_headway_20/matched"
    train_20_pt_control = PTOperator(train_20_gtfs_dir)

    train_30_gtfs_dir = r"data/gtfs/train/train_headway_30/matched"
    train_30_pt_control = PTOperator(train_30_gtfs_dir)
    print("PTOperator 数据加载完毕。")

    # --- 2. 定义所有场景参数 ---
    train_headway_list = [10, 20, 30]
    mod_fleet_size_list = [30, 50, 70, 90, 110, 130, 150]
    maas_communication_strategy_list = ['default', 'TPCS']
    random_seed_list = [3, 6, 9]
    mod_waiting_time_threshold_list = [300, 600, 900]
    mod_detour_time_threshold_list = [30, 60, 90]
    demand_size_list = [i for i in range(100, 1001, 100)]
    demand_split_ratio_list = [0, 20, 40, 60, 80]
    
    total_sim_time = [0, 10800]
    warmup_time = 3600
    time_period = [warmup_time, total_sim_time[1]+warmup_time]
    
    amod_request_level_analysis_folder = "data/amod-sim-results"
    demand_files_folder = "data/demand/11-500/amod"

    # --- 3. 创建所有场景组合 ---
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
    
    print(f"总共要处理的场景数量: {len(all_scenario_combinations)}")

    # --- 4. 设置 Multiprocessing Pool ---
    
    # 决定使用多少个工作进程 (例如：CPU 核心数 - 2，至少为 1)
    num_workers = max(1, cpu_count() - 2)
    print(f"启动一个包含 {num_workers} 个工作进程的 Pool...")

    # 准备要传递给 init_worker 的参数
    init_args = (
        walking_time_matrix,
        train_10_pt_control,
        train_20_pt_control,
        train_30_pt_control,
        demand_files_folder,
        amod_request_level_analysis_folder,
        STREET_STATION_TRANSFER_TIME,
        time_period
    )
    
    results = []
    
    # --- 5. 运行 Pool ---
    # 使用 'with' 语句来确保 Pool 被正确关闭
    start_time = time.time()
    with Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
        
        # pool.starmap 会将 all_scenario_combinations 中的每个元组解包
        # 作为参数传递给 process_scenario
        # 我们将它包装在 tqdm 中以显示进度条
        results = list(tqdm(pool.starmap(process_scenario, all_scenario_combinations), 
                            total=len(all_scenario_combinations),
                            desc="正在处理场景"))
    
    end_time = time.time()
    print(f"\n--- 处理完成，总耗时: {end_time - start_time:.2f} 秒 ---")
    
    # --- 6. 报告结果 ---
    errors = [r for r in results if r[1] == "ERROR"]
    skipped = [r for r in results if r[1] == "SKIPPED"]
    success = [r for r in results if r[1] == "SUCCESS"]
    
    print(f"成功处理: {len(success)}")
    print(f"跳过 (文件丢失): {len(skipped)}")
    print(f"错误: {len(errors)}")
    
    if skipped:
        print("\n--- 被跳过的场景 (最多显示 10 个) ---")
        for s in skipped[:10]:
            print(f"场景: {s[0]} | 原因: {s[2]}")
            
    if errors:
        print("\n--- 出错的场景 (最多显示 10 个) ---")
        for e in errors[:10]:
            print(f"场景: {e[0]} | 错误: {e[2]}")

# --- 主执行保护 ---
# 这对于 multiprocessing 在 Windows/macOS 上正确运行至关重要
if __name__ == "__main__":
    main()