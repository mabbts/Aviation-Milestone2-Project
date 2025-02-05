from datetime import datetime
from ..queries.state_vector_queries import StateVectorQueries
from ..retrieval.interval_generation import generate_chunk_intervals, generate_sample_intervals
from ..retrieval.retrieval_engine import retrieve_data_by_intervals
from ..utils.time_utils import parse_date

class StateVectorPipeline:
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
        """
        # Convert string dates to datetime objects.
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)

        # Generate intervals for the query.
        intervals = generate_chunk_intervals(start_dt, end_dt, chunk_hours)

        # Define query function using the StateVectorQueries.
        def query_fn(s, e):
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
        Retrieve state vector data by sampling n random intervals of length interval_hours
        between start_date and end_date.
        """
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)

        # Generate random sample intervals.
        intervals = generate_sample_intervals(start_dt, end_dt, n_samples, interval_hours)

        # Define query function.
        def query_fn(s, e):
            return StateVectorQueries.get_state_vectors(s, e)

        # Retrieve the sampled data.
        retrieve_data_by_intervals(intervals, query_fn, output_dir, prefix="state_vectors_sample", skip_if_exists=skip_if_exists) 