# flight_preprocessing.py
import pandas as pd
from pathlib import Path

def clean_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example transformation function that:
      - Strips whitespace from callsign
      - Fills missing airport codes with a default value
      - Drops rows with invalid timestamps
    """
    if df.empty:
        return df
    
    # Clean callsign
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].fillna("").str.strip()
    
    # Fill missing airport codes
    for col in ["estdepartureairport", "estarrivalairport", "airportofdeparture", "airportofdestination"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")
    
    # Drop rows with nonsensical times
    for time_col in ["firstseen", "lastseen", "takeofftime", "landingtime"]:
        if time_col in df.columns:
            df = df[df[time_col] > 0]

    # ... more cleaning or feature engineering

    return df

def transform_csv_file(input_csv: Path, output_csv: Path) -> None:
    """
    Reads a CSV file, applies cleaning/transformation steps, then writes to output_csv.
    """
    if not input_csv.exists():
        print(f"[flight_preprocessing] Input file does not exist: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    df_clean = clean_flight_data(df)

    # You could do further transformations, e.g., adding columns, grouping, etc.
    
    df_clean.to_csv(output_csv, index=False)
    print(f"[flight_preprocessing] Transformed data saved to {output_csv}")
