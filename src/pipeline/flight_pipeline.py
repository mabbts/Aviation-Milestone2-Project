"""
This module provides a pipeline for retrieving flight data.
It includes functions for retrieving data in standard chunks and by sampling.
"""

from datetime import datetime
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
