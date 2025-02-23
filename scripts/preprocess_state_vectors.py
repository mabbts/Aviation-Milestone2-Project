"""
Script to preprocess state vector data:
- Loads raw parquet files of state vectors from INPUT_DIR.
- Keeps only the columns of interest.
- Groups flight tracks by ICAO24 and performs linear interpolation (using time ordering).
- Ensures no missing data in the final processed columns.
- Saves the cleaned data to OUTPUT_DIR.
"""

from pathlib import Path
import polars as pl
import glob

# Constants - update these manually
INPUT_DIR = Path("data/raw/enid_data")
OUTPUT_DIR = Path("data/processed/enid_data")
OUTPUT_FILE = OUTPUT_DIR / "processed_state_vectors.parquet"

# Columns configuration
# We need these columns from the raw data for interpolation (time is used for sorting):
COLUMNS_TO_KEEP = ['icao24', 'time', 'lat', 'lon', 'velocity', 'vertrate', 'heading', 'geoaltitude']
# Only these numeric columns will be interpolated:
INTERPOLATE_COLUMNS = ['lat', 'lon', 'velocity', 'vertrate', 'heading', 'geoaltitude']
# Final processed output columns (dropping 'time'):
FINAL_COLUMNS = ['icao24', 'lat', 'lon', 'velocity', 'vertrate', 'heading', 'geoaltitude']


def load_data(input_dir: Path) -> pl.LazyFrame:
    """
    Loads all parquet files from the specified input directory into a Polars LazyFrame.
    """
    file_pattern = str(input_dir / "*.parquet")
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    
    # Use scan_parquet for lazy evaluation
    df = pl.scan_parquet(files)
    return df


def process_data(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Process the state vector data using Polars operations:
    - Selects only the required columns
    - Drops rows missing 'icao24' or 'time'
    - Groups by icao24 and interpolates missing numeric values
    - Drops any remaining missing values in the numeric columns
    - Explodes interpolated arrays back into rows
    """
    return (df
        # Select only necessary columns
        .select(COLUMNS_TO_KEEP)
        # Drop nulls in key columns
        .drop_nulls(['icao24', 'time'])
        # Sort by icao24 and time for interpolation
        .sort(['icao24', 'time'])
        # Group by icao24 and interpolate each numeric column
        .group_by('icao24')
        .agg([
            pl.col(col).interpolate() for col in INTERPOLATE_COLUMNS
        ])
        # Explode all columns except icao24 to convert arrays back to rows
        .explode(INTERPOLATE_COLUMNS)
        # Drop any remaining nulls in numeric columns
        .drop_nulls(INTERPOLATE_COLUMNS)
        # Select final columns
        .select(FINAL_COLUMNS))


def load_and_process_chunks(input_dir: Path, chunk_size: int = 30) -> None:
    """
    Loads and processes parquet files in chunks of specified size.
    Each chunk is saved as a separate file.
    """
    file_pattern = str(input_dir / "*.parquet")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    
    # Process chunks and save as separate files
    for i in range(0, len(files), chunk_size):
        chunk_files = files[i:i + chunk_size]
        chunk_number = i // chunk_size + 1
        output_file = OUTPUT_DIR / f"processed_state_vectors_chunk_{chunk_number}.parquet"
        
        print(f"Processing chunk {chunk_number} ({len(chunk_files)} files)...")
        
        (pl.scan_parquet(chunk_files)
            .pipe(process_data)
            .collect()
            .write_parquet(output_file, compression="snappy"))
        
        print(f"Saved chunk {chunk_number} to {output_file}")
    
    print("All chunks processed successfully!")


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starting chunked processing...")
    load_and_process_chunks(INPUT_DIR)
    print(f"Processed state vectors saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
