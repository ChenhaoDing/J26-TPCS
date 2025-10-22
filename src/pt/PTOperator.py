import os
import pandas as pd
from datetime import datetime
import ast
import typing as tp
import pandas as pd

try:
    from src.pt.cpp_pt_router.PyPTRouter import PyPTRouter
except ImportError as e:
    from cpp_pt_router.PyPTRouter import PyPTRouter

class PTOperator:
    def __init__(self, fp_gtfs_dir: str, print_logs: bool=True):
        self.print_logs = print_logs
        # initialize the pt router
        self.pt_router = None
        self._initialize_pt_router(fp_gtfs_dir)

        # load the stations and street station transfers from the gtfs data
        self.stations_fp_df = self._load_stations_from_gtfs(fp_gtfs_dir)
        self.street_station_transfers_fp_df = self._load_street_station_transfers_from_gtfs(fp_gtfs_dir)

        if self.print_logs:
            print("PT operator initialized successfully.")

    def _initialize_pt_router(self, fp_gtfs_dir: str,):
        """This method initializes the PT router.

        Args:
            fp_gtfs_dir (str): The directory containing the GTFS data of the operator.
        """
        # check if the directories exist and is all mandatory files present
        mandatory_files = [
            "agency_fp.txt", 
            "stops_fp.txt", 
            "trips_fp.txt", 
            "routes_fp.txt", 
            "calendar_fp.txt", 
            "stop_times_fp.txt", 
            "stations_fp.txt", 
            "transfers_fp.txt"
            ]
        if not os.path.exists(fp_gtfs_dir):
            raise FileNotFoundError(f"The directory {fp_gtfs_dir} does not exist.")
        for file in mandatory_files:
            if not os.path.exists(os.path.join(fp_gtfs_dir, file)):
                raise FileNotFoundError(f"The file {file} does not exist in the directory {fp_gtfs_dir}.")


        # initialize the pt router with the given gtfs data
        if self.print_logs:
            print(f"Initializing the PT router with the given GTFS data in the directory: {fp_gtfs_dir}")
        self.pt_router = PyPTRouter(fp_gtfs_dir)
        if self.print_logs:
            print("PT router initialized successfully.")

    def _load_stations_from_gtfs(self, fp_gtfs_dir: str) -> pd.DataFrame:
        """This method will load the stations from the GTFS data.

        Args:
            fp_gtfs_dir (str): The directory containing the GTFS data of the operator.

        Returns:
            pd.DataFrame: The PT stations data.
        """
        dtypes = {
            'station_id': 'str',
            'station_name': 'str',
            'station_lat': 'float',
            'station_lon': 'float',
            'stops_included': 'str',
            'station_stop_transfer_times': 'str',
            'num_stops_included': 'int',
        }
        return pd.read_csv(os.path.join(fp_gtfs_dir, "stations_fp.txt"), dtype=dtypes)
    
    def _load_street_station_transfers_from_gtfs(self, fp_gtfs_dir: str) -> pd.DataFrame:
        """This method will load the street station transfers from the GTFS data.

        Args:
            fp_gtfs_dir (str): The directory containing the GTFS data of the operator.

        Returns:
            pd.DataFrame: The transfer data between the street nodes and the pt stations.
        """
        dtypes = {
            'node_id': 'int',
            'closest_station_id': 'str',
            'street_transfer_time': 'int',
        }
        return pd.read_csv(os.path.join(fp_gtfs_dir, "street_station_transfers_fp.txt"), dtype=dtypes)
    
    def return_fastest_pt_journey_1to1(
        self,
        source_station_id: str,
        target_station_id: str,
        arrival_datetime: datetime,
        max_transfers: int=999,
        detailed: bool=False,
    ) -> tp.Union[tp.Dict[str, tp.Any], None]:
        """This method will return the fastest PT journey plan between two PT stations.
        A station may consist of multiple stops.

        Args:
            source_station_id (str): The id of the source station.
            target_station_id (str): The id of the target station.
            arrival_datetime (datetime): The arrival datetime at the source station.
            max_transfers (int): The maximum number of transfers allowed in the journey, 99 for no limit.
            detailed (bool): Whether to return the detailed journey plan.
        Returns:
            tp.Union[tp.Dict[str, tp.Any], None]: The fastest PT journey plan or None if no journey is found.
        """
        # get all included stops for the source and target station
        included_sources = self._get_included_stops_and_transfer_times(source_station_id)
        included_targets = self._get_included_stops_and_transfer_times(target_station_id)

        if self.print_logs:
            print(f"Finding fastest PT journey from station {source_station_id} to station {target_station_id} at {arrival_datetime} with max transfers {max_transfers} and detailed={detailed}")
        
        return self.pt_router.return_fastest_pt_journey_1to1(arrival_datetime, included_sources, included_targets, max_transfers, detailed)
    
    def return_fastest_pt_journey_xtox(
        self,
        source_station_ids: list[str],
        target_station_ids: list[str],
        arrival_datetime: datetime,  # all source stations share the same arrival time
        max_transfers: int=999,
        detailed: bool=False,
    ) -> tp.Union[tp.Dict[str, tp.Any], None]:
        
        # Call the router to find the fastest journey for each combination of source and target stations
        fastest_journeys = {}
        for source_station_id in source_station_ids:
            for target_station_id in target_station_ids:
                journey = self.return_fastest_pt_journey_1to1(
                    str(source_station_id),
                    str(target_station_id),
                    arrival_datetime,
                    max_transfers,
                    detailed
                )
                if journey:
                    fastest_journeys[(source_station_id, target_station_id)] = journey

        # Return the fastest journey found and corresponding source/target station IDs or None if no journey is found
        if fastest_journeys:
            # Find the overall fastest journey
            overall_fastest_journey = min(fastest_journeys.values(), key=lambda x: x['duration'])
            selected_source_station_id, selected_target_station_id = next(k for k, v in fastest_journeys.items() if v == overall_fastest_journey)
            return overall_fastest_journey, selected_source_station_id, selected_target_station_id
        return None
   
    def _get_included_stops_and_transfer_times(self, station_id: str) -> tp.Tuple[tp.List[str], tp.List[int]]:
        """This method will return the included stops and transfer times for a given station.

        Args:
            station_id (str): The id of the station.

        Returns:
            tp.Tuple[tp.List[str], tp.List[int]]: The included stops and transfer times.
        """
        station_data = self.stations_fp_df[self.stations_fp_df["station_id"] == station_id]
        
        if station_data.empty:
            raise ValueError(f"Station ID {station_id} not found in the stations data")
            
        included_ids_str = station_data["stops_included"].iloc[0]
        included_ids_str = included_ids_str.replace(';', ',')
        included_ids = ast.literal_eval(included_ids_str)

        transfer_times_str = station_data["station_stop_transfer_times"].iloc[0]
        transfer_times_str = transfer_times_str.replace(';', ',')
        transfer_times = ast.literal_eval(transfer_times_str)
        return [(stop_id, int(transfer_time)) for stop_id, transfer_time in zip(included_ids, transfer_times)]
    
    def find_closest_pt_station(self, street_node_id: int) -> tp.Tuple[str, int]:
        """This method finds the closest pt station from the street node id.

        Args:
            street_node_id (int): The street node id.

        Returns:
            tp.Tuple[str, int]: The closest pt station id and the walking time.
        """
        street_station_transfer = self.street_station_transfers_fp_df[self.street_station_transfers_fp_df["node_id"] == street_node_id]
        if street_station_transfer.empty:
            raise ValueError(f"Street node id {street_node_id} not found in the street station transfers file")
        closest_station_id: str = street_station_transfer["closest_station_id"].iloc[0]
        walking_time: int = street_station_transfer["street_station_transfer_time"].iloc[0]
        return closest_station_id, walking_time
    
    def find_closest_street_node(self, pt_station_id: str) -> tp.Tuple[int, int]:
        """This method finds the closest street node from the pt station id.

        Args:
            pt_station_id (str): The pt station id.

        Returns:
            tp.Tuple[int, int]: The closest street node id and the walking time.
        """
        street_station_transfers = self.street_station_transfers_fp_df[self.street_station_transfers_fp_df["closest_station_id"] == pt_station_id]
        if street_station_transfers.empty:
            raise ValueError(f"PT station id {pt_station_id} not found in the street station transfers file")
        # find the record with the minimum street_station_transfer_time
        min_transfer = street_station_transfers.loc[street_station_transfers["street_station_transfer_time"].idxmin()]
        closest_street_node_id: int = min_transfer["node_id"]
        walking_time: int = min_transfer["street_station_transfer_time"]
        return closest_street_node_id, walking_time
    

if __name__ == "__main__":
    # Test the pt control class： python -m src.pt.PTControlBasicCpp
    example_gtfs_dir = "data/GTFS/example"
    pt_control = PTOperator(example_gtfs_dir)
    arrival_datetime = datetime(2024, 1, 1, 0, 4, 0)
    import time
    start_time = time.time()
    print(pt_control.return_fastest_pt_journey_1to1("s1", "s14", arrival_datetime, 1, detailed=False))
    print(f"Time taken: {time.time() - start_time} seconds")
    start_time = time.time()
    print(pt_control.return_fastest_pt_journey_1to1("s1", "s14", arrival_datetime, 1, detailed=True))
    print(f"Time taken: {time.time() - start_time} seconds")