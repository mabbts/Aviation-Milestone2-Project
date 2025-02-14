#!/usr/bin/env python3
"""
Script to fetch state vectors for each accident flight listed in the
'accident_flights_icao.parquet' file. For each row (flight), this script:
  - Loads the accident flight data from a parquet file (containing icao24, firstseen, lastseen).
  - Uses the ICAO24 and time interval to build a SQL query via StateVectorQueries.get_state_vectors_by_icao.
  - Executes the query using a Trino client.
  - Saves the returned state vector data to a parquet file (one per flight) in the same directory.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.queries.state_vector_queries import StateVectorQueries
from src.utils.paths import DATA_DIR
from pyopensky.trino import Trino

def main():
    # Define the path to the accident flights parquet file
    input_file = DATA_DIR / "processed" / "accident_flights_icao.parquet"
    
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    print(f"[INFO] Loading accident flights data from: {input_file}")
    try:
        flights_df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"[ERROR] Failed to load parquet file: {e}")
        sys.exit(1)
    
    # Validate that the expected columns exist
    expected_columns = {'icao24', 'firstseen', 'lastseen'}
    if not expected_columns.issubset(set(flights_df.columns)):
        print(f"[ERROR] Input file is missing expected columns. Expected columns: {expected_columns}")
        sys.exit(1)
    
    # Initialize the Trino client for query execution
    trino = Trino()

    # Define the output directory (same as the input file directory)
    output_dir = input_file.parent
    print(f"[INFO] Query results will be saved in: {output_dir}")

    # Loop over each flight (row) and execute the state vector query
    for idx, row in flights_df.iterrows():
        icao = row['icao24']
        # Convert the timestamps to integers if necessary.
        firstseen = int(row['firstseen'])
        lastseen = int(row['lastseen'])

        print(f"[INFO] Querying state vectors for flight (ICAO24: {icao}, Firstseen: {firstseen}, Lastseen: {lastseen})")
        
        # create a buffer of 10 minutes around the firstseen and lastseen timestamps
        firstseen_buffer = firstseen - 10 * 60
        lastseen_buffer = lastseen + 10 * 60

        # Build the SQL query using the provided ICAO24 and time interval.
        query = StateVectorQueries.get_state_vectors_by_icao(icao, start_time=firstseen_buffer, end_time=lastseen_buffer)
        print(f"[DEBUG] Generated query:\n{query}")
        
        try:
            result_df = trino.query(query)
            # Check if the query returned any data.
            if result_df is None or (hasattr(result_df, '__len__') and len(result_df) == 0):
                print(f"[WARNING] No state vector data returned for ICAO {icao} in the specified interval.")
                continue
        except Exception as e:
            print(f"[ERROR] Query execution failed for ICAO {icao}: {e}")
            continue
        
        # Create an output file name using the ICAO24 and timestamps for uniqueness.
        output_filename = f"state_vectors_{icao}_{firstseen}_{lastseen}.parquet"
        output_file = output_dir / output_filename
        
        try:
            result_df.to_parquet(output_file, index=False)
            print(f"[SUCCESS] Saved data for ICAO {icao} to: {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save data for ICAO {icao} to file: {e}")
    
    print("[INFO] All queries have been processed.")

if __name__ == "__main__":
    main() 