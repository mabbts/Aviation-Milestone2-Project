#!/usr/bin/env python3
"""
Script to run the get_flight_data_by_icao query from flights_data4
using the Trino client and save the resulting flight data in a parquet file
within a new folder under the specified data_dir.
"""

import os
import sys
import datetime
from pyopensky.trino import Trino  # using Trino for query execution
from pathlib import Path
# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# Adjust import if your project structure is different.
from src.queries.flight_queries import FlightQueries
from src.utils.paths import DATA_DIR

def main():
    # ----- Configuration -----
    # ICAO24 list to query
    icao = ['a8a7d1', 'a11dde', 'a46188']
    
    # Define the base directory where the output folder will be created.
    data_dir = DATA_DIR / "raw/icao_flights"
    # -------------------------

    print(f"[START] Fetching flight data for ICAO: {icao}")

    # Generate the SQL query for this ICAO.
    query = FlightQueries.get_flight_data_by_icao(icao)
    print(f"[INFO] Generated SQL query:\n{query}")

    # Initialize Trino client and execute the query.
    trino = Trino()
    try:
        print("[INFO] Executing query using Trino...")
        # Add debug mode to get more information just in case
        trino.debug = True
        df = trino.query(query)
        
        if df is None:
            print("[WARNING] Query returned None instead of a DataFrame")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Error executing query with Trino: {e}")
        print("[DEBUG] Error type:", type(e))
        print("[DEBUG] Full error details:", str(e))
        sys.exit(1)

    # Check if any data was returned.
    num_rows = len(df) if hasattr(df, "__len__") else 0
    if num_rows == 0:
        print("[INFO] Query executed successfully, but no data was returned.")
        sys.exit(0)
    else:
        print(f"[INFO] Query returned {num_rows} rows.")

    # Ensure the base output directory exists.
    os.makedirs(data_dir, exist_ok=True)
    
    # Define the fixed parquet file path.
    parquet_file = os.path.join(data_dir, "flight_data.parquet")
    try:
        print(f"[INFO] Saving results to parquet file: {parquet_file}")
        # Save the DataFrame to a parquet file without the index.
        df.to_parquet(parquet_file, index=False)
    except Exception as e:
        print(f"[ERROR] Error writing parquet file: {e}")
        sys.exit(1)
    
    print(f"[SUCCESS] Query executed and results saved in {parquet_file}")
    
    if __name__ == "__main__":
        main()
