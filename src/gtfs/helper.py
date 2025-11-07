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


def create_basic_bus_schedule(
    bb_stop_time, bb_hub_stop_time, bb_transfer_stop_time, 
    TRAIN_ARRIVAL_TIME, TRAIN_DEPARTURE_TIME, BUS_DRIVE_TIME
):
    bb_hub_arrival_time = (TRAIN_ARRIVAL_TIME + TRAIN_DEPARTURE_TIME) / 2 - bb_hub_stop_time / 2
    bb_hub_departure_time = bb_hub_arrival_time + bb_hub_stop_time

    bb_117_departure_time = bb_hub_arrival_time - BUS_DRIVE_TIME 
    bb_117_arrival_time = bb_117_departure_time - bb_stop_time

    bb_57_departure_time = bb_117_arrival_time - BUS_DRIVE_TIME
    bb_57_arrival_time = bb_57_departure_time - bb_stop_time

    bb_56_departure_time = bb_57_arrival_time - BUS_DRIVE_TIME
    bb_56_arrival_time = bb_56_departure_time - bb_transfer_stop_time

    bb_55_departure_time = bb_56_arrival_time - BUS_DRIVE_TIME
    bb_55_arrival_time = bb_55_departure_time - bb_stop_time

    bb_54_departure_time = bb_55_arrival_time - BUS_DRIVE_TIME

    bb_118_arrival_time = bb_hub_departure_time + BUS_DRIVE_TIME
    bb_118_departure_time = bb_118_arrival_time + bb_stop_time

    bb_58_arrival_time = bb_118_departure_time + BUS_DRIVE_TIME
    bb_58_departure_time = bb_58_arrival_time + bb_stop_time

    bb_59_arrival_time = bb_58_departure_time + BUS_DRIVE_TIME
    bb_59_departure_time = bb_59_arrival_time + bb_transfer_stop_time

    bb_60_arrival_time = bb_59_departure_time + BUS_DRIVE_TIME
    bb_60_departure_time = bb_60_arrival_time + bb_stop_time

    bb_61_arrival_time = bb_60_departure_time + BUS_DRIVE_TIME

    arrival_times = [
        bb_55_arrival_time, bb_56_arrival_time, bb_57_arrival_time, 
        bb_117_arrival_time, bb_hub_arrival_time, bb_118_arrival_time, 
        bb_58_arrival_time, bb_59_arrival_time, bb_60_arrival_time, bb_61_arrival_time
    ]

    departure_times = [
        bb_54_departure_time, bb_55_departure_time, bb_56_departure_time, bb_57_departure_time,
        bb_117_departure_time, bb_hub_departure_time, bb_118_departure_time, 
        bb_58_departure_time, bb_59_departure_time, bb_60_departure_time, 
    ]

    # Convert seconds to HH:MM:SS format
    arrival_times = [str(timedelta(seconds=int(at))) for at in arrival_times]
    departure_times = [str(timedelta(seconds=int(dt))) for dt in departure_times]
    
    return arrival_times, departure_times

def create_diagonal_bus_schedule(
    bb_stop_time, bb_hub_stop_time,
    TRAIN_ARRIVAL_TIME, TRAIN_DEPARTURE_TIME, BUS_DRIVE_TIME
):
    bb_hub_arrival_time = (TRAIN_ARRIVAL_TIME + TRAIN_DEPARTURE_TIME) / 2 - bb_hub_stop_time / 2
    bb_hub_departure_time = bb_hub_arrival_time + bb_hub_stop_time

    bb_119_departure_time = bb_hub_arrival_time - BUS_DRIVE_TIME 
    bb_119_arrival_time = bb_119_departure_time - bb_stop_time

    bb_66_departure_time = bb_119_arrival_time - BUS_DRIVE_TIME
    bb_66_arrival_time = bb_66_departure_time - bb_stop_time

    bb_76_departure_time = bb_66_arrival_time - BUS_DRIVE_TIME
    bb_76_arrival_time = bb_76_departure_time - bb_stop_time

    bb_75_departure_time = bb_76_arrival_time - BUS_DRIVE_TIME
    bb_75_arrival_time = bb_75_departure_time - bb_stop_time

    bb_86_departure_time = bb_75_arrival_time - BUS_DRIVE_TIME
    bb_86_arrival_time = bb_86_departure_time - bb_stop_time

    bb_85_departure_time = bb_86_arrival_time - BUS_DRIVE_TIME
    bb_85_arrival_time = bb_85_departure_time - bb_stop_time

    bb_96_departure_time = bb_85_arrival_time - BUS_DRIVE_TIME
    bb_96_arrival_time = bb_96_departure_time - bb_stop_time

    bb_95_departure_time = bb_96_arrival_time - BUS_DRIVE_TIME
    bb_95_arrival_time = bb_95_departure_time - bb_stop_time

    bb_105_departure_time = bb_95_arrival_time - BUS_DRIVE_TIME

    bb_116_arrival_time = bb_hub_departure_time + BUS_DRIVE_TIME
    bb_116_departure_time = bb_116_arrival_time + bb_stop_time

    bb_49_arrival_time = bb_116_departure_time + BUS_DRIVE_TIME
    bb_49_departure_time = bb_49_arrival_time + bb_stop_time

    bb_39_arrival_time = bb_49_departure_time + BUS_DRIVE_TIME
    bb_39_departure_time = bb_39_arrival_time + bb_stop_time

    bb_40_arrival_time = bb_39_departure_time + BUS_DRIVE_TIME
    bb_40_departure_time = bb_40_arrival_time + bb_stop_time

    bb_29_arrival_time = bb_40_departure_time + BUS_DRIVE_TIME
    bb_29_departure_time = bb_29_arrival_time + bb_stop_time

    bb_30_arrival_time = bb_29_departure_time + BUS_DRIVE_TIME
    bb_30_departure_time = bb_30_arrival_time + bb_stop_time

    bb_19_arrival_time = bb_30_departure_time + BUS_DRIVE_TIME
    bb_19_departure_time = bb_19_arrival_time + bb_stop_time

    bb_20_arrival_time = bb_19_departure_time + BUS_DRIVE_TIME
    bb_20_departure_time = bb_20_arrival_time + bb_stop_time

    bb_9_arrival_time = bb_20_departure_time + BUS_DRIVE_TIME
    bb_9_departure_time = bb_9_arrival_time + bb_stop_time

    bb_10_arrival_time = bb_9_departure_time + BUS_DRIVE_TIME

    arrival_times = [
        bb_95_arrival_time, bb_96_arrival_time, bb_85_arrival_time, bb_86_arrival_time,
        bb_75_arrival_time, bb_76_arrival_time, bb_66_arrival_time, bb_119_arrival_time,
        bb_hub_arrival_time, bb_116_arrival_time, bb_49_arrival_time, bb_39_arrival_time,
        bb_40_arrival_time, bb_29_arrival_time, bb_30_arrival_time, bb_19_arrival_time,
        bb_20_arrival_time, bb_9_arrival_time, bb_10_arrival_time
    ]

    departure_times = [
        bb_105_departure_time,
        bb_95_departure_time, bb_96_departure_time, bb_85_departure_time, bb_86_departure_time,
        bb_75_departure_time, bb_76_departure_time, bb_66_departure_time, bb_119_departure_time,
        bb_hub_departure_time, bb_116_departure_time, bb_49_departure_time, bb_39_departure_time,
        bb_40_departure_time, bb_29_departure_time, bb_30_departure_time, bb_19_departure_time,
        bb_20_departure_time, bb_9_departure_time
    ]

    # Convert seconds to HH:MM:SS format
    arrival_times = [str(timedelta(seconds=int(at))) for at in arrival_times]
    departure_times = [str(timedelta(seconds=int(dt))) for dt in departure_times]

    return arrival_times, departure_times


def create_ring_bus_schedule(rb_56_departure_time, rb_stop_time, rb_transfer_stop_time, BUS_DRIVE_TIME):
    rb_64_arrival_time = rb_56_departure_time + BUS_DRIVE_TIME
    rb_64_departure_time = rb_64_arrival_time + rb_stop_time

    rb_74_arrival_time = rb_64_departure_time + BUS_DRIVE_TIME
    rb_74_departure_time = rb_74_arrival_time + rb_stop_time

    rb_85_arrival_time = rb_74_departure_time + BUS_DRIVE_TIME
    rb_85_departure_time = rb_85_arrival_time + rb_stop_time

    rb_86_arrival_time = rb_85_departure_time + BUS_DRIVE_TIME
    rb_86_departure_time = rb_86_arrival_time + rb_stop_time

    rb_87_arrival_time = rb_86_departure_time + BUS_DRIVE_TIME
    rb_87_departure_time = rb_87_arrival_time + rb_stop_time

    rb_88_arrival_time = rb_87_departure_time + BUS_DRIVE_TIME
    rb_88_departure_time = rb_88_arrival_time + rb_transfer_stop_time

    rb_89_arrival_time = rb_88_departure_time + BUS_DRIVE_TIME
    rb_89_departure_time = rb_89_arrival_time + rb_stop_time

    rb_90_arrival_time = rb_89_departure_time + BUS_DRIVE_TIME
    rb_90_departure_time = rb_90_arrival_time + rb_stop_time

    rb_91_arrival_time = rb_90_departure_time + BUS_DRIVE_TIME
    rb_91_departure_time = rb_91_arrival_time + rb_stop_time

    rb_80_arrival_time = rb_91_departure_time + BUS_DRIVE_TIME
    rb_80_departure_time = rb_80_arrival_time + rb_stop_time

    rb_69_arrival_time = rb_80_departure_time + BUS_DRIVE_TIME
    rb_69_departure_time = rb_69_arrival_time + rb_stop_time

    rb_59_arrival_time = rb_69_departure_time + BUS_DRIVE_TIME
    rb_59_departure_time = rb_59_arrival_time + rb_transfer_stop_time

    rb_51_arrival_time = rb_59_departure_time + BUS_DRIVE_TIME
    rb_51_departure_time = rb_51_arrival_time + rb_stop_time

    rb_41_arrival_time = rb_51_departure_time + BUS_DRIVE_TIME
    rb_41_departure_time = rb_41_arrival_time + rb_stop_time

    rb_30_arrival_time = rb_41_departure_time + BUS_DRIVE_TIME
    rb_30_departure_time = rb_30_arrival_time + rb_stop_time

    rb_29_arrival_time = rb_30_departure_time + BUS_DRIVE_TIME
    rb_29_departure_time = rb_29_arrival_time + rb_stop_time

    rb_28_arrival_time = rb_29_departure_time + BUS_DRIVE_TIME
    rb_28_departure_time = rb_28_arrival_time + rb_stop_time

    rb_27_arrival_time = rb_28_departure_time + BUS_DRIVE_TIME
    rb_27_departure_time = rb_27_arrival_time + rb_transfer_stop_time

    rb_26_arrival_time = rb_27_departure_time + BUS_DRIVE_TIME
    rb_26_departure_time = rb_26_arrival_time + rb_stop_time

    rb_25_arrival_time = rb_26_departure_time + BUS_DRIVE_TIME
    rb_25_departure_time = rb_25_arrival_time + rb_stop_time

    rb_24_arrival_time = rb_25_departure_time + BUS_DRIVE_TIME
    rb_24_departure_time = rb_24_arrival_time + rb_stop_time

    rb_35_arrival_time = rb_24_departure_time + BUS_DRIVE_TIME
    rb_35_departure_time = rb_35_arrival_time + rb_stop_time

    rb_46_arrival_time = rb_35_departure_time + BUS_DRIVE_TIME
    rb_46_departure_time = rb_46_arrival_time + rb_stop_time

    rb_56_arrival_time = rb_46_departure_time + BUS_DRIVE_TIME

    arrival_times = [
        rb_64_arrival_time, rb_74_arrival_time, rb_85_arrival_time,
        rb_86_arrival_time, rb_87_arrival_time, rb_88_arrival_time, rb_89_arrival_time,
        rb_90_arrival_time, rb_91_arrival_time, rb_80_arrival_time, rb_69_arrival_time,
        rb_59_arrival_time, rb_51_arrival_time, rb_41_arrival_time, rb_30_arrival_time,
        rb_29_arrival_time, rb_28_arrival_time, rb_27_arrival_time, rb_26_arrival_time,
        rb_25_arrival_time, rb_24_arrival_time, rb_35_arrival_time, rb_46_arrival_time,rb_56_arrival_time
    ]

    departure_times = [
        rb_56_departure_time, rb_64_departure_time, rb_74_departure_time, rb_85_departure_time,
        rb_86_departure_time, rb_87_departure_time, rb_88_departure_time, rb_89_departure_time,
        rb_90_departure_time, rb_91_departure_time, rb_80_departure_time, rb_69_departure_time,
        rb_59_departure_time, rb_51_departure_time, rb_41_departure_time, rb_30_departure_time,
        rb_29_departure_time, rb_28_departure_time, rb_27_departure_time, rb_26_departure_time,
        rb_25_departure_time, rb_24_departure_time, rb_35_departure_time, rb_46_departure_time,
    ]

    # Convert seconds to HH:MM:SS format
    arrival_times = [str(timedelta(seconds=int(at))) for at in arrival_times]
    departure_times = [str(timedelta(seconds=int(dt))) for dt in departure_times]

    return arrival_times, departure_times