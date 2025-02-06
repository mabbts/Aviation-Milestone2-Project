import pandas as pd
import glob 
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from src.transformations.flight_preprocessing import preprocess_flight_data
from src.utils.paths import DATA_DIR


def main():
    # Set the input file here.
    # Update the path below to the actual path for your flight data parquet files.
    input_dir_path = DATA_DIR / "raw" / "v4_samples"

    # Get all parquet files in the input directory.
    parquet_files = glob.glob(str(input_dir_path / "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {input_dir_path}")

    n_samples = 20
    parquet_files = parquet_files[:n_samples]
    
    # Define the output directory: DATA_DIR/processed/
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the same filename as input for the output file.
    output_path = processed_dir / "v4_samples.parquet"
    
    print(f"Reading flight data from {input_dir_path} ...")
    
    # Read and concatenate all parquet files
    dfs = []
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Process the DataFrame to compute new track metrics.
    df_processed = preprocess_flight_data(df)
    
    # Optionally, drop the original 'track' column if no longer needed.
    if "track" in df_processed.columns:
        df_processed.drop(columns=["track"], inplace=True)
    
    # Save the processed data.
    df_processed.to_parquet(output_path, index=False)
    print(f"Processed flight data saved to {output_path}")

if __name__ == '__main__':
    main()