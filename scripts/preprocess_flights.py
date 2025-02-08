"""
This script preprocesses flight data by reading parquet files,
concatenating them into chunks, applying transformations, and
saving the processed chunks to a specified directory.
"""

import pandas as pd
import glob
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from src.transformations.flight_preprocessing import preprocess_flight_data
from src.utils.paths import DATA_DIR


def main():
    """
    Main function to orchestrate the flight data preprocessing.
    """
    # Set the input directory: update as needed.
    input_dir_path = DATA_DIR / "raw" / "v4_samples"

    # Get all parquet files in the input directory.
    parquet_files = glob.glob(str(input_dir_path / "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {input_dir_path}")

    # Define chunk size (20 files per batch)
    chunk_size = 20
    chunks = [parquet_files[i:i + chunk_size] for i in range(0, len(parquet_files), chunk_size)]
    print(f"Processing data in {len(chunks)} chunks.")

    # Set processed output directory.
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Define whether to drop the 'track' column
    drop_track = True

    # Process each chunk separately.
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx + 1} with {len(chunk)} files...")

        # Read and concatenate all files in the current chunk.
        dfs = []
        for parquet_file in chunk:
            df = pd.read_parquet(parquet_file)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        # Process the DataFrame to compute new track metrics.
        df_processed = preprocess_flight_data(df)

        # Optionally, drop the original 'track' column if no longer needed.
        if drop_track and "track" in df_processed.columns:
            df_processed.drop(columns=["track"], inplace=True)

        # Save the processed data with a chunk-specific filename.
        output_path = processed_dir / f"v4_samples_chunk_{idx + 1}.parquet"
        df_processed.to_parquet(output_path, index=False)
        print(f"Processed chunk {idx + 1} saved to {output_path}")


if __name__ == '__main__':
    main()