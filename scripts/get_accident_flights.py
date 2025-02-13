#!/usr/bin/env python3
"""
Script to retrieve flight data for all ICAO numbers and dates from the accident_icao24_dates.csv file
using batch processing to handle the large number of queries efficiently.
"""

import os
import sys
import csv
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.pipeline.flight_pipeline import FlightsPipeline
from src.utils.paths import DATA_DIR

def main():
    # Define the output directory for the flight data
    output_dir = DATA_DIR / "raw/accident_flights_2"
    
    # Read all ICAO numbers and dates from the CSV file
    icao_date_list = []
    csv_path = DATA_DIR / "accident_icao24_dates.csv"
    
    print(f"[START] Reading ICAO data from {csv_path}")
    
    try:
        with open(csv_path, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                icao_date_list.append((row[0], row[1]))
    except FileNotFoundError:
        print(f"[ERROR] Could not find the CSV file at {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {e}")
        sys.exit(1)

    print(f"[INFO] Found {len(icao_date_list)} ICAO entries to process")

    # Process all entries using the batch method
    try:
        FlightsPipeline.batch_flight_by_icao(
            icao_date_list=icao_date_list,
            output_dir=str(output_dir),
            batch_size=50,
            skip_if_exists=True
        )
    except Exception as e:
        print(f"[ERROR] Failed during batch processing: {e}")
        sys.exit(1)

    print(f"[SUCCESS] Processing complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main() 