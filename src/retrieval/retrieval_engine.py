"""
This module provides a generic data retrieval function that retrieves data over a list of time intervals,
using a provided query function, and saves the results to parquet files.
"""

import os
from pathlib import Path
import pandas as pd
from pyopensky.trino import Trino
from src.utils.file_utils import maybe_save_parquet

def retrieve_data_by_intervals(
    intervals: list[tuple[int, int]],
    query_fn,
    output_dir: str,
    prefix: str,
    skip_if_exists: bool = True
):
    """
    Retrieve data over a list of time intervals using a provided query function and save results to parquet files.

    Args:
        intervals (list[tuple[int, int]]): A list of tuples, where each tuple contains the start and end timestamps (Unix epoch time) of an interval.
        query_fn (function): A function that accepts a start time and end time (both as integers) and returns an SQL query string.
        output_dir (str): The directory where the parquet files should be written.
        prefix (str): The filename prefix for the parquet files (e.g., "flight_v4").
        skip_if_exists (bool, optional): If True, skip retrieval for intervals that already have a non-empty parquet file. Defaults to True.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trino = Trino()

    for (start_ts, end_ts) in intervals:
        file_name = f"{prefix}_{start_ts}_{end_ts}.parquet"
        parquet_file = output_path / file_name
        
        print(f"[retrieve_data_by_intervals] Processing interval {start_ts} to {end_ts}")
        query = query_fn(start_ts, end_ts)
        print(f"[retrieve_data_by_intervals] Executing query: {query}")
        
        try:
            df = trino.query(query)
            print(f"[retrieve_data_by_intervals] Query returned {len(df)} rows")
            
            if maybe_save_parquet(df, parquet_file, skip_if_exists):
                print(f"[retrieve_data_by_intervals] Wrote {file_name} ({len(df)} rows).")
            else:
                print(f"[retrieve_data_by_intervals] Skipped or no data for {file_name}.")
        except Exception as e:
            print(f"[retrieve_data_by_intervals] Error retrieving {file_name}: {e}")
