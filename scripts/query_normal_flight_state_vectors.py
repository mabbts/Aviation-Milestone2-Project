#!/usr/bin/env python3
"""
Script to fetch state vectors for a random sample (200 flights) of normal flights
from the global flights dataset. For each sampled flight, this script:
  - Loads global flights data from a parquet file.
  - Randomly samples 200 rows.
  - Converts datetime columns to UNIX timestamps.
  - Uses the flight's ICAO24 and a buffered time interval to build a SQL query.
  - Executes the query using a Trino client.
  - Saves the returned state vector data to a parquet file (one file per flight)
    in the DATA_DIR/raw/normal_flight_states directory.
"""

import sys
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.queries.state_vector_queries import StateVectorQueries
from src.utils.paths import DATA_DIR
from pyopensky.trino import Trino

def main():
    # Define the path to the global flights parquet file
    parquet_file = DATA_DIR / "processed" / "global_flights" / "v4_samples_chunk_3.parquet"
    
    if not parquet_file.exists():
        print(f"[ERROR] Input file not found: {parquet_file}")
        sys.exit(1)

    print(f"[INFO] Loading global flights data from: {parquet_file}")
    
    try:
        df = dd.read_parquet(str(parquet_file))
    except Exception as e:
        print(f"[ERROR] Failed to read parquet file: {e}")
        sys.exit(1)
    
    # Compute total rows and calculate the fraction for sampling 200 rows
    try:
        total_rows = df.shape[0].compute()
        if total_rows == 0:
            print("[ERROR] No rows found in the dataset.")
            sys.exit(1)
        frac = 200 / total_rows
    except Exception as e:
        print(f"[ERROR] Failed to compute total number of rows: {e}")
        sys.exit(1)
    
    try:
        sample = df.sample(frac=frac, random_state=42).compute()
    except Exception as e:
        print(f"[ERROR] Failed to sample rows: {e}")
        sys.exit(1)
    
    # Select only the essential columns: 'icao24', 'firstseen', 'lastseen'
    try:
        sample = sample[['icao24', 'firstseen', 'lastseen']]
    except KeyError as e:
        print(f"[ERROR] Expected column not found in the dataset: {e}")
        sys.exit(1)
    
    # Convert 'firstseen' and 'lastseen' from datetime to UNIX timestamp if necessary
    try:
        if pd.api.types.is_datetime64_any_dtype(sample['firstseen']):
            sample['firstseen'] = sample['firstseen'].apply(lambda dt: int(dt.timestamp()))
        if pd.api.types.is_datetime64_any_dtype(sample['lastseen']):
            sample['lastseen'] = sample['lastseen'].apply(lambda dt: int(dt.timestamp()))
    except Exception as e:
        print(f"[ERROR] Failed to convert datetime to UNIX timestamps: {e}")
        sys.exit(1)
    
    print(f"[INFO] Sampled {len(sample)} flights from global flights data.")
    
    # Define the output directory for state vector files
    output_dir = DATA_DIR / "raw" / "normal_flight_states"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Query results will be saved in: {output_dir}")
    
    # Initialize the Trino client for query execution
    trino = Trino()

    # Loop over each sampled flight and execute the state vector query
    for idx, row in sample.iterrows():
        icao = row['icao24']
        try:
            firstseen = int(row['firstseen'])
            lastseen = int(row['lastseen'])
        except Exception as e:
            print(f"[ERROR] Failed to parse timestamps for ICAO {icao}: {e}")
            continue
        
        print(f"[INFO] Querying state vectors for flight (ICAO24: {icao}, Firstseen: {firstseen}, Lastseen: {lastseen})")
        
        # Create a 10-minute buffer (600 seconds) around the flight timestamps
        firstseen_buffer = firstseen - 10 * 60
        lastseen_buffer = lastseen + 10 * 60
        
        # Build the SQL query using the buffered interval
        query = StateVectorQueries.get_state_vectors_by_icao(icao, start_time=firstseen_buffer, end_time=lastseen_buffer)
        print(f"[DEBUG] Generated query:\n{query}")
        
        try:
            result_df = trino.query(query)
            # Check if the query returned any data
            if result_df is None or (hasattr(result_df, '__len__') and len(result_df) == 0):
                print(f"[WARNING] No state vector data returned for ICAO {icao} in the specified interval.")
                continue
        except Exception as e:
            print(f"[ERROR] Query execution failed for ICAO {icao}: {e}")
            continue
        
        # Create an output filename and save the results to a parquet file
        output_filename = f"state_vectors_{icao}_{firstseen}_{lastseen}.parquet"
        output_file = output_dir / output_filename
        
        try:
            result_df.to_parquet(output_file, index=False)
            print(f"[SUCCESS] Saved state vector data for ICAO {icao} to: {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save data for ICAO {icao} to file: {e}")
    
    print("[INFO] All queries have been processed.")

if __name__ == "__main__":
    main()