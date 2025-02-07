# retrieval_engine.py
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
    Generic data retrieval over a list of intervals.
    
    intervals: list of (start_timestamp, end_timestamp)
    query_fn: function that takes (start_time, end_time) -> SQL query string
    output_dir: where parquet files should be written
    prefix: filename prefix (e.g. "flight_v4")
    skip_if_exists: if True, skip intervals that already have a non-empty parquet file
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
