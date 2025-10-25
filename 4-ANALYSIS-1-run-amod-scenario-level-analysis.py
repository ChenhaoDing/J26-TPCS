import pandas as pd
import os
from tqdm import tqdm
from itertools import product
import multiprocessing
from functools import partial
import sys

from src.analysis.amod import process_amod_scenario_level_stats

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

output_folder = r"data\amod-sim-results"
os.makedirs(output_folder, exist_ok=True)

def process_scenario(scenario_params, amod_fleetpy_simulation_output_folder, time_period):
    (random_seed, fleet_size, demand_size, demand_split_ratio, 
        broker_type, op_max_detour_time_factor, op_max_wait_time, headway) = scenario_params

    scenario_name = f"amod-{demand_size}-{demand_split_ratio}-{fleet_size}-{broker_type}-{op_max_detour_time_factor}-{op_max_wait_time}-{headway}-{random_seed}-{time_period[0]}-{time_period[1]}"

    amod_simulation_results_dir = os.path.join(amod_fleetpy_simulation_output_folder, scenario_name)

    request_level_stats_df = pd.read_csv(os.path.join(output_folder, scenario_name, "amod_request_level_analysis_results.csv"))

    results_dict = process_amod_scenario_level_stats(
        request_level_stats_df = request_level_stats_df,
        amod_simulation_results_dir = amod_simulation_results_dir
    )

    if results_dict is None or not results_dict:
        print(f"There are no results for {scenario_name}", file=sys.stderr)
        return None
    
    results_dict['random_seed'] = random_seed
    results_dict['fleet_size'] = fleet_size 
    results_dict['demand_size'] = demand_size
    results_dict['demand_split_ratio'] = demand_split_ratio
    results_dict['broker_type'] = broker_type
    results_dict['op_max_detour_time_factor'] = op_max_detour_time_factor
    results_dict['op_max_wait_time'] = op_max_wait_time
    results_dict['headway'] = headway
    results_dict['scenario_name'] = scenario_name

    return results_dict

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

    all_results_dicts = []

    num_processes = os.cpu_count()
    if num_processes:
        num_processes = max(1, num_processes - 3)
    else:
        num_processes = 4

    print(f"--- Starting processing with {num_processes} worker processes... ---")

    worker_func = partial(
        process_scenario,
        amod_fleetpy_simulation_output_folder=amod_fleetpy_simulation_output_folder,
        time_period=time_period
    )

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterable = pool.imap_unordered(worker_func, all_scenario_combinations)

        for results_dict in tqdm(results_iterable, total=len(all_scenario_combinations), desc="Processing scenarios"):
            if results_dict is not None and results_dict:
                # Save individual scenario results
                scenario_name = results_dict['scenario_name']
                scenario_output_dirpath = os.path.join(output_folder, scenario_name)
                os.makedirs(scenario_output_dirpath, exist_ok=True)
                scenario_output_filepath = os.path.join(scenario_output_dirpath, "amod_scenario_level_analysis_results.csv")
                pd.DataFrame([results_dict]).to_csv(scenario_output_filepath, index=False)

                all_results_dicts.append(results_dict)

    print("\n--- ...Processing complete ---")

    print("Merging all results...")
    final_results_df = pd.DataFrame(all_results_dicts)

    print("Final results DataFrame shape:", final_results_df.shape)

    final_results_df_file_path = os.path.join(output_folder, "all_amod_scenario_level_analysis_results.csv")
    final_results_df.to_csv(final_results_df_file_path, index=False)
    print(f"Final results saved to: {final_results_df_file_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
