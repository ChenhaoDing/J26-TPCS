import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from itertools import product
import multiprocessing
from functools import partial

from src.analysis.amod import process_amod_user_stats

# Variables
# Train Headway (min)
train_headway_list = [i for i in range(10, 31, 5)]

# MoD Fleet Size
mod_fleet_size_list = [i for i in range(30, 151, 20)]

# MaaS Platform Communication Strategy
maas_communication_strategy_list = ['default', 'TPCS']

# Random Seed
random_seed_list = [3, 6, 9]

# MoD Waiting Time Threshold (s)
mod_waiting_time_threshold_list = [300, 600, 900]
# MoD Detour Time Threshold (%)
mod_detour_time_threshold_list = [i for i in range(30, 101, 20)]

# Demand Size
demand_size_list = [i for i in range(100, 600, 50)]
# Demand Split Ratio (Intra Modal, %)
demand_split_ratio_list = [i for i in range(10, 91, 20)]

# Total Simulation Time (s)
total_sim_time = [0, 10800]  # 3 hours
# Warm-up Time (s)
warmup_time = 3600  # 1 hour
# Simulation Time Period (s)
time_period = [warmup_time, total_sim_time[1] + warmup_time]  # 1h + 3h

amod_fleetpy_simulation_output_folder = r"D:\projects\fleetpy\github\ptbroker\studies\j26-tpcs\results"

demand_files_folder = r"data\demand\11-500\amod"

output_folder = r"data\amod-sim-results"
os.makedirs(output_folder, exist_ok=True)

def process_scenario(scenario_params, demand_files_folder, amod_fleetpy_simulation_output_folder, time_period):
    try:
        (random_seed, fleet_size, demand_size, demand_split_ratio, 
         broker_type, op_max_detour_time_factor, op_max_wait_time, headway) = scenario_params

        scenario_name = f"amod-{demand_size}-{demand_split_ratio}-{fleet_size}-{broker_type}-{op_max_detour_time_factor}-{op_max_wait_time}-{headway}-{random_seed}-{time_period[0]}-{time_period[1]}"

        demand_name = f"amod_ds{demand_size}_dsr{demand_split_ratio}_rs{random_seed}.csv"
        demand_file_path = os.path.join(demand_files_folder, demand_name)
        demand_df = pd.read_csv(demand_file_path)

        user_stats_file_path = os.path.join(amod_fleetpy_simulation_output_folder, scenario_name, "1_user-stats.csv")
        if not os.path.exists(user_stats_file_path):
            print(f"The file is not found: {user_stats_file_path}")
            return None
            
        user_stats_df = pd.read_csv(user_stats_file_path)

        results_df = process_amod_user_stats(user_stats_df, demand_df)

        if results_df is None or results_df.empty:
            print(f"There are no results for {scenario_name}")
            return None

        results_df['random_seed'] = random_seed
        results_df['fleet_size'] = fleet_size
        results_df['demand_size'] = demand_size
        results_df['demand_split_ratio'] = demand_split_ratio
        results_df['broker_type'] = broker_type
        results_df['op_max_detour_time_factor'] = op_max_detour_time_factor
        results_df['op_max_wait_time'] = op_max_wait_time
        results_df['headway'] = headway
        results_df['scenario_name'] = scenario_name

        return results_df

    except FileNotFoundError as e:
        print(f"File not found (processing {scenario_params}): {e}")
        return None
    except Exception as e:
        print(f"Error occurred (processing {scenario_params}): {e}")
        return None

def main():
    # Request level
    # All scenario combinations
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

    print(f"There are {len(all_scenario_combinations)} scenarios to process.")

    all_results_dfs = []

    num_processes = os.cpu_count()
    if num_processes:
        num_processes = max(1, num_processes - 3)
    else:
        num_processes = 4

    print(f"--- Starting processing with {num_processes} worker processes... ---")

    worker_func = partial(process_scenario,
                          demand_files_folder=demand_files_folder,
                          amod_fleetpy_simulation_output_folder=amod_fleetpy_simulation_output_folder,
                          time_period=time_period)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterable = pool.imap_unordered(worker_func, all_scenario_combinations)

        for result_df in tqdm(results_iterable, total=len(all_scenario_combinations), desc="Processing scenarios"):
            if result_df is not None and not result_df.empty:
                # Save individual scenario results
                scenario_name = result_df['scenario_name'].iloc[0]
                scenario_output_dirpath = os.path.join(output_folder, scenario_name)
                os.makedirs(scenario_output_dirpath, exist_ok=True)
                scenario_output_filepath = os.path.join(scenario_output_dirpath, "amod_request_level_analysis_results.csv")
                result_df.to_csv(scenario_output_filepath, index=False)
                all_results_dfs.append(result_df)

    print("\n--- ...Processing complete ---")

    print("Merging all results...")
    final_results_df = pd.concat(all_results_dfs, ignore_index=True)
    final_results_df['unique_request_id'] = final_results_df.index

    print("Final results DataFrame shape:", final_results_df.shape)
    
    final_results_df_file_path = os.path.join(output_folder, "all_amod_request_level_analysis_results.csv")
    final_results_df.to_csv(final_results_df_file_path, index=False)
    print(f"Final results saved to: {final_results_df_file_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
