# flights_pipeline.py
from datetime import datetime
from ..queries.flight_queries import FlightQueries
from ..retrieval.interval_generation import (
    generate_chunk_intervals,
    generate_sample_intervals
)
from ..retrieval.retrieval_engine import retrieve_data_by_intervals
from ..utils.time_utils import parse_date

class FlightsPipeline:

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
        """
        # Convert string to datetime
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)

        # Generate intervals
        intervals = generate_chunk_intervals(start_dt, end_dt, chunk_hours)

        # Define query_fn
        def query_fn(s, e):
            return FlightQueries.get_flight_data_v4(s, e)

        # Retrieve data
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
        Retrieve flight data (v4) by sampling n random intervals of length interval_hours
        between start_date and end_date.
        """
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)

        # Generate intervals
        intervals = generate_sample_intervals(start_dt, end_dt, n_samples, interval_hours)

        def query_fn(s, e):
            return FlightQueries.get_flight_data_v4(s, e)

        retrieve_data_by_intervals(intervals, query_fn, output_dir, prefix="flight_v4_sample", skip_if_exists=skip_if_exists)
