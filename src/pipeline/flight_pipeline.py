"""
This module provides a pipeline for retrieving flight data.
It includes functions for retrieving data in standard chunks and by sampling.
"""

from datetime import datetime
from typing import List
from ..queries.flight_queries import FlightQueries
from ..retrieval.interval_generation import (
    generate_chunk_intervals,
    generate_sample_intervals
)
from ..retrieval.retrieval_engine import retrieve_data_by_intervals
from ..utils.time_utils import parse_date

class FlightsPipeline:
    """
    Pipeline for retrieving flight data.
    """

    @staticmethod
    def chunked_flight_v4(
        start_date: str,
        end_date: str,
        output_dir: str,
        chunk_hours: float = 1.0,
        skip_if_exists: bool = True
    ):
        """
        Retrieve flight data (v4) in standard chunks between start_date and end_date.

        Args:
            start_date (str): Start date for data retrieval (e.g., 'YYYY-MM-DD HH:MM:SS').
            end_date (str): End date for data retrieval (e.g., 'YYYY-MM-DD HH:MM:SS').
            output_dir (str): Directory to save the retrieved data chunks.
            chunk_hours (float): Duration of each chunk in hours. Defaults to 1.0.
            skip_if_exists (bool): If True, skip retrieval if the file already exists. Defaults to True.
        """
        # Convert string to datetime objects
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)

        # Generate time intervals for each chunk
        intervals = generate_chunk_intervals(start_dt, end_dt, chunk_hours)

        # Define the query function to retrieve flight data for a given time interval
        def query_fn(s, e):
            """Retrieves flight data v4 for the given start and end datetimes."""
            return FlightQueries.get_flight_data_v4(s, e)

        # Retrieve data for each interval and save it to the output directory
        retrieve_data_by_intervals(intervals, query_fn, output_dir, prefix="flight_v4", skip_if_exists=skip_if_exists)

    @staticmethod
    def sample_flight_v4(
        start_date: str,
        end_date: str,
        output_dir: str,
        n_samples: int,
        interval_hours: float = 24.0,
        skip_if_exists: bool = True
    ):
        """
        Retrieve flight data (v4) by sampling n random intervals between start_date and end_date.
        If interval_hours > 1, each sampled interval is further subdivided (chunked) into 1-hour intervals,
        and data is queried hour by hour.

        Args:
            start_date (str): Start date, e.g., in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.
            end_date (str): End date in the same format as start_date.
            output_dir (str): Directory where the parquet files should be stored.
            n_samples (int): Number of sample intervals to generate.
            interval_hours (float): The total duration of each sample interval in hours.
                                    If greater than 1, the interval will be chunked into 1-hour segments.
            skip_if_exists (bool): If True, skip writing if the output file already exists.
        """
        # Parse start and end dates into datetime objects
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)

        # Generate random sample intervals
        sample_intervals = generate_sample_intervals(start_dt, end_dt, n_samples, interval_hours)

        # If interval_hours > 1, further chunk each sample interval into 1-hour pieces.
        if interval_hours > 1:
            intervals = []
            for start_ts, end_ts in sample_intervals:
                dt_start = datetime.fromtimestamp(start_ts)
                dt_end = datetime.fromtimestamp(end_ts)
                hourly_intervals = generate_chunk_intervals(dt_start, dt_end, chunk_hours=1.0)
                intervals.extend(hourly_intervals)
        else:
            intervals = sample_intervals

        # Define query function to retrieve flight data for a given time interval
        def query_fn(s, e):
            """Retrieves flight data v4 for the given start and end datetimes."""
            return FlightQueries.get_flight_data_v4(s, e)

        # Retrieve the sampled data
        retrieve_data_by_intervals(
            intervals,
            query_fn,
            output_dir,
            prefix="flight_v4_sample",
            skip_if_exists=skip_if_exists
        )

    @staticmethod
    def batch_flight_by_icao(icao_date_list: list, output_dir: str, batch_size: int = 100, skip_if_exists: bool = True):
        """
        Retrieve flight data for a large list of ICAO identifiers and their associated dates by batching them into queries.
        Each batch handles at most batch_size ICAO identifiers and computes the appropriate time range with buffer.

        Args:
            icao_date_list (list): List of tuples containing (icao, date) pairs.
            output_dir (str): Directory where the parquet files will be saved.
            batch_size (int): Maximum number of ICAO identifiers per batch query. Defaults to 100.
            skip_if_exists (bool): If True, skip the batch if the output file already exists.
        """
        import os
        from datetime import datetime, timedelta
        from pyopensky.trino import Trino

        # Ensure the output directory exists.
        os.makedirs(output_dir, exist_ok=True)

        total_icaos = len(icao_date_list)
        total_batches = (total_icaos + batch_size - 1) // batch_size
        print(f"[INFO] Processing {total_icaos} ICAOs in {total_batches} batches (up to {batch_size} per batch).")

        trino = Trino()
        trino.debug = True

        for batch_index in range(total_batches):
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, total_icaos)
            batch_data = icao_date_list[start_index:end_index]

            output_file = os.path.join(output_dir, f"flight_by_icao_batch_{batch_index}.parquet")
            if skip_if_exists and os.path.exists(output_file):
                print(f"[INFO] Skipping batch {batch_index} (output file {output_file} already exists).")
                continue

            print(f"[INFO] Processing batch {batch_index+1}/{total_batches}: ICAOs {start_index} to {end_index-1}")

            # Extract ICAO numbers and compute time range for this batch
            batch_icaos = [item[0] for item in batch_data]
            
            # Convert dates and add buffer
            base_min_time = min(datetime.fromisoformat(item[1]) for item in batch_data)
            base_max_time = max(datetime.fromisoformat(item[1]) for item in batch_data)
            
            # Add 3-hour buffer before and after
            min_time = int((base_min_time - timedelta(hours=3)).timestamp())
            max_time = int((base_max_time + timedelta(hours=8)).timestamp())

            print(f"[INFO] Time range for batch {batch_index+1}: {min_time} to {max_time}")

            # Generate the SQL query for the current batch with time range
            query = FlightQueries.get_flight_data_by_icao(batch_icaos, time_range=(min_time, max_time))
            print(f"[DEBUG] Executing query for batch {batch_index+1}:\n{query}")

            try:
                df = trino.query(query)
                if df is None:
                    print(f"[WARNING] Batch {batch_index+1}: Query returned None, skipping batch.")
                    continue
            except Exception as e:
                print(f"[ERROR] Failed to execute query for batch {batch_index+1}: {e}")
                continue

            try:
                # Save the DataFrame to a parquet file without the index.
                df.to_parquet(output_file, index=False)
                print(f"[SUCCESS] Batch {batch_index+1}: Saved {len(df)} rows to {output_file}")
            except Exception as e:
                print(f"[ERROR] Failed to save output for batch {batch_index+1}: {e}")
