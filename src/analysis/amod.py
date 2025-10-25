import pandas as pd
import numpy as np
import os
import re
import sys
import glob

MAIN_DIR = r'D:\projects\papers\J26-TPCS'
sys.path.append(MAIN_DIR)
from src.misc.globals import *

DEFAULT_BOARDING_TIME = 30

EURO_PER_TON_OF_CO2 = 145 # from BVWP2030 Modulhandbuch (page 113)
EMISSION_CPG = 145 * 100 / 1000**2
ENERGY_EMISSIONS = 112 # g/kWh from https://www.swm.de/dam/swm/dokumente/geschaeftskunden/broschuere-strom-erdgas-gk.pdf
PV_G_CO2_KM = 130 # g/km from https://www.ris-muenchen.de/RII/RII/DOK/ANTRAG/2337762.pdf with 60:38 benzin vs diesel

def process_amod_user_stats(
        user_stats_df: pd.DataFrame, 
        demand_df: pd.DataFrame, 
        default_boarding_time: int = DEFAULT_BOARDING_TIME
    ) -> pd.DataFrame:
    """Process AMoD user statistics at the request level.
    KPIs:
    1. Total journey time
    2. Served by AMoD
    3. FM served
    4. PT waiting time
    5. LM served
    """
    # Prepare output csv file
    results_list = []

    # Get all request IDs
    request_ids = user_stats_df['request_id'].unique()
    for request_id in request_ids:
        request_data = user_stats_df[user_stats_df['request_id'] == request_id]
        parent_request_data = request_data[request_data['is_parent_request'] == 1].iloc[0]

        demand_data = demand_df[demand_df['request_id'] == request_id].iloc[0]

        rq_type = demand_data['rq_type']
        subnetwork = demand_data['subnetwork']
        request_time = demand_data['rq_time']

        if rq_type == 'intra': # pure amod request
            dropoff_time = parent_request_data['dropoff_time']
            if not np.isnan(dropoff_time):
                total_journey_time = dropoff_time - request_time + default_boarding_time
                served_by_amod = 1
            else:
                total_journey_time = np.nan
                served_by_amod = 0
            fm_served = -1
            pt_waiting_time = -1
            lm_served = -1

        else: # intermodal request
            fm_request_data = request_data[request_data['sub_trip_id'] == 5].iloc[0]
            pt_request_data = request_data[request_data['sub_trip_id'] == 6].iloc[0]
            lm_request_data = request_data[request_data['sub_trip_id'] == 7].iloc[0]

            # FM
            fm_dropoff_time = fm_request_data['dropoff_time']
            if not np.isnan(fm_dropoff_time):
                fm_served = 1

                # PT
                pt_offer_str = pt_request_data['offers']
                pt_waiting_time = _extract_pt_waiting_time(pt_offer_str)

            else:
                fm_served = 0
                pt_waiting_time = np.nan

            # LM
            lm_dropoff_time = lm_request_data['dropoff_time']
            if not np.isnan(lm_dropoff_time):
                lm_served = 1
            else:
                lm_served = 0
            
            # Request level KPIs
            if fm_served == 1 and lm_served == 1:
                total_journey_time = lm_dropoff_time - request_time + default_boarding_time
                served_by_amod = 1
            else:
                total_journey_time = np.nan
                served_by_amod = 0
        
        # Append to results dataframe
        new_row = {
            "request_id": request_id,
            "total_journey_time": total_journey_time,
            "served_by_amod": served_by_amod,
            "fm_served": fm_served,
            "pt_waiting_time": pt_waiting_time,
            "lm_served": lm_served,
            "rq_type": rq_type,
            "subnetwork": subnetwork
        }
        results_list.append(new_row)
    results_df = pd.DataFrame(results_list)

    return results_df

def _extract_pt_waiting_time(offer_str: str) -> int:
    """Extract PT waiting time from offers string."""
    pt_waiting_time = int(re.search(r't_wait:(\d+)', offer_str).group(1))
    return pt_waiting_time

def _read_op_output_file(output_dir, op_id, evaluation_start_time = None, evaluation_end_time = None) -> pd.DataFrame:
    """ this method reads the ouputfile for the operator and returns its dataframe
    :param output_dir: directory of the scenario results
    :param op_id: operator id to evaluate
    :param evaluation_start_time: if given all entries starting before this time are discarded
    :param evaluation_end_time: if given, all entries starting after this time are discarded
    :return: output dataframe of specific operator
    """
    op_df = pd.read_csv(os.path.join(output_dir, f"2-{int(op_id)}_op-stats.csv"))
    if evaluation_start_time is not None:
        op_df = op_df[op_df[G_VR_LEG_START_TIME] >= evaluation_start_time]
    if evaluation_end_time is not None:
        op_df = op_df[op_df[G_VR_LEG_START_TIME] < evaluation_end_time]
    # test for correct datatypes
    def convert_str(val):
        if val != val:
            return val
        if type(val) == str:
            return val
        else:
            return str(int(val))
    test_convert = [G_VR_ALIGHTING_RID, G_VR_BOARDING_RID, G_VR_OB_RID]
    for col in test_convert:
        if op_df.dtypes[col] != str:
            op_df[col] = op_df[col].apply(convert_str)
    return op_df

def _create_vehicle_type_db(vehicle_data_dir):
    list_veh_data_f = glob.glob(f"{vehicle_data_dir}/*csv")
    veh_type_db = {}    # veh_type -> veh_type_data
    for f in list_veh_data_f:
        veh_type_name = os.path.basename(f)[:-4]
        veh_type_data = pd.read_csv(f, index_col=0).squeeze("columns")
        veh_type_db[veh_type_name] = {}
        for k, v in veh_type_data.items():
            try:
                veh_type_db[veh_type_name][k] = float(v)
            except:
                veh_type_db[veh_type_name][k] = v
        veh_type_db[veh_type_name][G_VTYPE_NAME] = veh_type_data.name
    return veh_type_db

def process_amod_scenario_level_stats(
        request_level_stats_df: pd.DataFrame,
        amod_simulation_results_dir: str,
    ) -> dict:
    """Process AMoD user statistics at the scenario level.
    KPIs:
    1. Average journey time
    2. Average AMoD service rate
    3. Average PT waiting time
    4. Average occupancy
    5. Total kilometers driven
    6. Average kilometers driven
    7. Total AMoD cost
    8. Average AMoD cost
    """
    scenario_parameters, list_operator_attributes, _ = load_scenario_inputs(amod_simulation_results_dir)
    dir_names = get_directory_dict(scenario_parameters, list_operator_attributes, abs_fleetpy_dir=r"D:\projects\fleetpy\github\ptbroker")

    results_dict = {}

    # Process AMoD operator stats
    amod_op_id = 0
    amod_operator_results_dict = _process_amod_scenario_level_amod_operator_stats(
        amod_simulation_results_dir,
        amod_op_id,
        scenario_parameters,
        dir_names
    )
    results_dict.update(amod_operator_results_dict)

    # Process request level stats
    amod_request_results_dict = _process_amod_scenario_level_amod_request_stats(
        request_level_stats_df
    )
    results_dict.update(amod_request_results_dict)
    return results_dict

def _process_amod_scenario_level_amod_operator_stats(
        amod_simulation_results_dir: str,
        amod_op_id: int,
        scenario_parameters: dict,
        dir_names: dict
    ) -> dict:
    results_dict = {}
    # ---------------------------- AMoDoperator --------------------------------
    # vehicle type data
    veh_type_db = _create_vehicle_type_db(dir_names[G_DIR_VEH])
    veh_type_stats = pd.read_csv(os.path.join(amod_simulation_results_dir, "2_vehicle_types.csv"))

    try:
        op_vehicle_df = _read_op_output_file(amod_simulation_results_dir, amod_op_id)
    except FileNotFoundError:
        op_vehicle_df = pd.DataFrame([], columns=[G_V_OP_ID, G_V_VID, G_VR_STATUS, G_VR_LOCKED, G_VR_LEG_START_TIME,
                                                G_VR_LEG_END_TIME, G_VR_LEG_START_POS, G_VR_LEG_END_POS,
                                                G_VR_LEG_DISTANCE, G_VR_LEG_START_SOC, G_VR_LEG_END_SOC,
                                                G_VR_TOLL, G_VR_OB_RID, G_VR_BOARDING_RID, G_VR_ALIGHTING_RID,
                                                G_VR_NODE_LIST, G_VR_REPLAY_ROUTE])

    n_vehicles = veh_type_stats[veh_type_stats[G_V_OP_ID]==amod_op_id].shape[0]
    sim_end_time = scenario_parameters["end_time"]
    simulation_time = scenario_parameters["end_time"] - scenario_parameters["start_time"]

    op_vehicle_df["VRL_end_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_END_TIME], sim_end_time)
    op_vehicle_df["VRL_start_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_START_TIME], sim_end_time)
    utilized_veh_df = op_vehicle_df[(op_vehicle_df["status"] != VRL_STATES.OUT_OF_SERVICE.display_name) & (op_vehicle_df["status"] != VRL_STATES.CHARGING.display_name)]
    utilization_time = utilized_veh_df["VRL_end_sim_end_time"].sum() - utilized_veh_df["VRL_start_sim_end_time"].sum()
    unutilized_veh_df = op_vehicle_df[(op_vehicle_df["status"] == VRL_STATES.OUT_OF_SERVICE.display_name) | (op_vehicle_df["status"] == VRL_STATES.CHARGING.display_name)]
    unutilized_time = unutilized_veh_df["VRL_end_sim_end_time"].sum() - unutilized_veh_df["VRL_start_sim_end_time"].sum()

    op_fleet_utilization = 100 * (utilization_time/(n_vehicles * simulation_time - unutilized_time))
    op_total_km = op_vehicle_df[G_VR_LEG_DISTANCE].sum()/1000.0
    
    # by vehicle stats
    # ----------------
    op_veh_types = veh_type_stats[veh_type_stats[G_V_OP_ID] == amod_op_id]
    op_veh_types.set_index(G_V_VID, inplace=True)
    all_vid_dict = {}
    for vid, vid_vtype_row in op_veh_types.iterrows():
        vtype_data = veh_type_db[vid_vtype_row[G_V_TYPE]]
        op_vid_vehicle_df = op_vehicle_df[op_vehicle_df[G_V_VID] == vid]
        veh_km = op_vid_vehicle_df[G_VR_LEG_DISTANCE].sum() / 1000
        veh_kWh = veh_km * vtype_data[G_VTYPE_BATTERY_SIZE] / vtype_data[G_VTYPE_RANGE]
        co2_per_kWh = scenario_parameters.get(G_ENERGY_EMISSIONS, ENERGY_EMISSIONS)
        if co2_per_kWh is None:
            co2_per_kWh = ENERGY_EMISSIONS
        veh_co2 = co2_per_kWh * veh_kWh
        veh_fix_costs = np.rint(scenario_parameters.get(G_OP_SHARE_FC, 1.0) * vtype_data[G_VTYPE_FIX_COST])
        veh_var_costs = np.rint(vtype_data[G_VTYPE_DIST_COST] * veh_km)
        # TODO # after ISTTT: idle times
        all_vid_dict[vid] = {"type":vtype_data[G_VTYPE_NAME], "total km":veh_km, "total kWh": veh_kWh,
                            "total CO2 [g]": veh_co2, "fix costs": veh_fix_costs,
                            "total variable costs": veh_var_costs}
    all_vid_df = pd.DataFrame.from_dict(all_vid_dict, orient="index")
    try:
        op_co2 = all_vid_df["total CO2 [g]"].sum()
        op_ext_em_costs = np.rint(EMISSION_CPG * op_co2)
        op_fix_costs = all_vid_df["fix costs"].sum()
        op_var_costs = all_vid_df["total variable costs"].sum()
    except:
        op_co2 = 0
        op_ext_em_costs = 0
        op_fix_costs = 0
        op_var_costs = 0

    def weight_ob_rq(entries):
        if pd.isnull(entries[G_VR_OB_RID]):
            return 0.0
        else:
            number_ob_rq = len(str(entries[G_VR_OB_RID]).split(";"))
            return number_ob_rq * entries[G_VR_LEG_DISTANCE]

    def weight_ob_pax(entries):
        try:
            return entries[G_VR_NR_PAX] * entries[G_VR_LEG_DISTANCE]
        except:
            return 0.0

    op_vehicle_df["weighted_ob_rq"] = op_vehicle_df.apply(weight_ob_rq, axis = 1)
    op_vehicle_df["weighted_ob_pax"] = op_vehicle_df.apply(weight_ob_pax, axis=1)
    op_distance_avg_rq = op_vehicle_df["weighted_ob_rq"].sum() / op_vehicle_df[G_VR_LEG_DISTANCE].sum()
    op_distance_avg_occupancy = op_vehicle_df["weighted_ob_pax"].sum() / op_vehicle_df[G_VR_LEG_DISTANCE].sum()

    results_dict["fleet_utilization_rate"] = op_fleet_utilization
    results_dict["distance_avg_occupancy"] = op_distance_avg_occupancy
    results_dict["total_km_driven"] = op_total_km
    results_dict["average_km_driven"] = op_total_km / n_vehicles
    results_dict["total_amod_cost"] = op_fix_costs + op_var_costs
    results_dict["average_amod_cost"] = (op_fix_costs + op_var_costs) / n_vehicles
    results_dict['total_amod_fix_cost'] = op_fix_costs
    results_dict['total_amod_variable_cost'] = op_var_costs
    results_dict["total_co2_emissions_g"] = op_co2
    results_dict["total_external_emission_costs"] = op_ext_em_costs

    return results_dict

def _process_amod_scenario_level_amod_request_stats(request_level_stats_df: pd.DataFrame) -> dict:
    """Process AMoD user statistics at the scenario level.
    KPIs:
    1. Average journey time
    2. Average AMoD service rate
    3. Average PT waiting time
    """
    results_dict = {}

    num_requests = request_level_stats_df.shape[0]
    served_requests_df = request_level_stats_df[request_level_stats_df["served_by_amod"] == 1]
    num_served_requests = served_requests_df.shape[0]

    # Average AMoD service rate
    average_amod_service_rate = num_served_requests / num_requests if num_requests > 0 else 0.0
    results_dict["average_amod_service_rate"] = average_amod_service_rate

    # Average journey time
    average_journey_time = served_requests_df["total_journey_time"].mean()
    results_dict["average_journey_time"] = average_journey_time

    # Average PT waiting time
    average_pt_waiting_time = served_requests_df["pt_waiting_time"].mean()
    results_dict["average_pt_waiting_time"] = average_pt_waiting_time

    return results_dict