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
    result_dict['avg_journey_time'] = scenario_results_df['best_journey_duration'].mean()

    # 2. Average PT service rate
    total_requests = len(scenario_results_df)
    pt_requests = scenario_results_df[scenario_results_df['best_option'] == 'pt']
    result_dict['avg_pt_service_rate'] = pt_requests.shape[0] / total_requests if total_requests > 0 else 0

    # Add parameters to results
    result_dict.update(scenario_parameters)

    return result_dict
