import pandas as pd
import numpy as np
import os

def process_pt_scenario_level_stats(
    scenario_results_df: pd.DataFrame,
    scenario_parameters: dict,
) -> dict:
    """
    Process public transport scenario-level statistics.
    1. Average journey time
    2. Average PT service rate
    *3. Total bus cost
    *4. Average bus cost
    """
    result_dict = {}

    # 1. Average journey time
    result_dict['average_journey_time'] = scenario_results_df['best_journey_duration'].mean()

    # 2. Average PT service rate
    total_requests = len(scenario_results_df)
    pt_requests = scenario_results_df[scenario_results_df['best_option'] == 'pt']
    result_dict['average_pt_service_rate'] = pt_requests.shape[0] / total_requests if total_requests > 0 else 0

    # 3. Average intra-city journey time
    intra_city_requests = scenario_results_df[scenario_results_df['rq_type'] == 'intra']
    result_dict['average_intracity_journey_time'] = intra_city_requests['best_journey_duration'].mean() if not intra_city_requests.empty else np.nan

    # 4. Average inter-city journey time
    inter_city_requests = scenario_results_df[scenario_results_df['rq_type'] == 'inter']
    result_dict['average_intercity_journey_time'] = inter_city_requests['best_journey_duration'].mean() if not inter_city_requests.empty else np.nan

    # Add parameters to results
    result_dict.update(scenario_parameters)

    return result_dict
