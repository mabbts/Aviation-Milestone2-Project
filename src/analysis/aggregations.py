from datetime import datetime
from pyopensky.trino import Trino
import pandas as pd
from ..database.queries import OpenSkyQueries
from ..utils.constants import GEORGIA_BOUNDS
from pathlib import Path
import os

def get_flight_aggregates(start_date: str, end_date: str) -> pd.DataFrame:
    """Get aggregated flight data for a date range"""
    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_time = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    
    trino = Trino()
    query = OpenSkyQueries.get_flight_aggregate(
        start_time=start_time,
        end_time=end_time,
        bounds=GEORGIA_BOUNDS
    )
    
    return trino.query(query) 

def get_detailed_flights(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get detailed flight data including tracks and durations for a date range.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame containing flight details including duration and track data
    """
    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_time = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    
    trino = Trino()
    query = OpenSkyQueries.get_detailed_flight_data(
        start_time=start_time,
        end_time=end_time,
    )
    
    return trino.query(query) 

def get_flight_data_chunks(start_date: str, end_date: str, output_dir: str) -> None:
    """
    Retrieve flight data in hourly chunks and save each chunk as a CSV file in the specified output directory.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.
        end_date: End date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.
        output_dir: Directory where each hourly CSV file will be saved.
        
    The function writes each hourly query result to a CSV file named as:
        flight_<interval_start>_<interval_end>.csv
    """


    def parse_date(date_str: str) -> datetime:
        """Helper function to parse a date string with or without time component."""
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Date format for {date_str} is not supported. Please use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.")

    # Convert dates to datetime objects using the helper function
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)

    # Create the output directory if it doesn't exist
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Initialize Trino connection
    trino = Trino()

    current_dt = start_dt
    while current_dt < end_dt:
        interval_start = int(current_dt.timestamp())
        # Move to the next hour
        current_dt += pd.Timedelta(hours=1)
        interval_end = int(current_dt.timestamp())

        # Define the file path for the current interval
        chunk_file = output_directory / f"flight_{interval_start}_{interval_end}.csv"
        
        # Skip this interval if the file already exists and is not empty
        if chunk_file.exists() and os.path.getsize(chunk_file) > 0:
            print(f"Skipping interval {interval_start} to {interval_end} as file exists.")
            continue

        # Get the query for the current interval
        query = OpenSkyQueries.get_flight_data_v4(
            start_time=interval_start,
            end_time=interval_end,
        )

        # Execute the query and write the results to disk
        try:
            df = trino.query(query)
            if not df.empty:
                df.to_csv(chunk_file, index=False)
                print(f"Wrote chunk for interval {interval_start} to {interval_end} into {chunk_file}.")
            else:
                print(f"No data returned for interval {interval_start} to {interval_end}.")
        except Exception as e:
            print(f"Error processing interval {interval_start} to {interval_end}: {e}")