from datetime import datetime
from ..queries.state_vector_queries import StateVectorQueries
from ..retrieval.interval_generation import generate_chunk_intervals, generate_sample_intervals
from ..retrieval.retrieval_engine import retrieve_data_by_intervals
from ..utils.time_utils import parse_date

class StateVectorPipeline:
    """
    Pipeline for retrieving state vector data.
    """
    @staticmethod
    def chunked_state_vectors(
        start_date: str,
        end_date: str,
        output_dir: str,
        chunk_hours: float = 1.0,
        skip_if_exists: bool = True
    ):
        """
        Retrieve state vector data in standard chunks between start_date and end_date.

        Args:
            start_date (str): Start date for data retrieval (e.g., 'YYYY-MM-DD HH:MM:SS').
            end_date (str): End date for data retrieval (e.g., 'YYYY-MM-DD HH:MM:SS').
            output_dir (str): Directory to save the retrieved data chunks.
            chunk_hours (float): Duration of each chunk in hours. Defaults to 1.0.
            skip_if_exists (bool): If True, skip retrieval if the file already exists. Defaults to True.
        """
        # Convert string dates to datetime objects.
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)

        # Generate intervals for the query.
        intervals = generate_chunk_intervals(start_dt, end_dt, chunk_hours)

        # Define query function using the StateVectorQueries.
        def query_fn(s, e):
            """Retrieves state vector data for the given start and end datetimes."""
            return StateVectorQueries.get_state_vectors(s, e)

        # Retrieve data using the generated intervals.
        retrieve_data_by_intervals(intervals, query_fn, output_dir, prefix="state_vectors", skip_if_exists=skip_if_exists)

    @staticmethod
    def sample_state_vectors(
        start_date: str,
        end_date: str,
        output_dir: str,
        n_samples: int,
        interval_hours: float = 24.0,
        skip_if_exists: bool = True
    ):
        """
        Retrieve state vector data by sampling n random intervals between start_date and end_date.
        If interval_hours > 1, each sampled interval is further subdivided (chunked) into 1-hour intervals,
        and data is queried hour by hour.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.
            end_date (str): End date in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.
            output_dir (str): Directory where the parquet files should be stored.
            n_samples (int): Number of sample intervals to generate.
            interval_hours (float): The total hours duration of each sample interval.
                                     If greater than 1, the interval will be chunked into 1-hour segments.
            skip_if_exists (bool): If True, skip writing if the output file already exists.
        """
        # Parse start and end dates into datetime objects
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)

        # Generate random sample intervals.
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

        # Define query function.
        def query_fn(s, e):
            """Retrieves state vector data for the given start and end datetimes."""
            return StateVectorQueries.get_state_vectors(s, e)

        # Retrieve the sampled data.
        retrieve_data_by_intervals(
            intervals,
            query_fn,
            output_dir,
            prefix="state_vectors_sample",
            skip_if_exists=skip_if_exists
        )