# file_utils.py
import os
from pathlib import Path
import pandas as pd

def maybe_save_csv(df: pd.DataFrame, file_path: Path, skip_if_exists=True) -> bool:
    """
    Save DataFrame to CSV if it doesn't already exist, or if skip_if_exists is False.
    Returns True if data was saved, False if it was skipped or df is empty.
    """
    if skip_if_exists and file_path.exists() and file_path.stat().st_size > 0:
        print(f"[file_utils] Skipping file {file_path}, already exists.")
        return False
    
    if not df.empty:
        df.to_csv(file_path, index=False)
        print(f"[file_utils] Saved data to {file_path}")
        return True
    else:
        print(f"[file_utils] No data to save for {file_path}")
        return False

def maybe_save_parquet(df: pd.DataFrame, file_path: Path, skip_if_exists=True) -> bool:
    """
    Save DataFrame to parquet if it doesn't already exist, or if skip_if_exists is False.
    Returns True if data was saved, False if it was skipped or df is empty.
    """
    if skip_if_exists and file_path.exists() and file_path.stat().st_size > 0:
        print(f"[file_utils] Skipping file {file_path}, already exists.")
        return False
    
    if not df.empty:
        df.to_parquet(file_path, index=False)
        print(f"[file_utils] Saved data to {file_path}")
        return True
    else:
        print(f"[file_utils] No data to save for {file_path}")
        return False
