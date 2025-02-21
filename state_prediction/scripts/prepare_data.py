#!/usr/bin/env python
"""
Script: prepare_data.py
-----------------------
Loads raw flight state vectors, resamples/interpolates them,
creates sliding windows for (X,y), splits, scales, and saves
train/test data + scalers to disk.
"""

import os
import glob
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from paths import DATA_DIR

def resample_flight_state_data(df, interval='5s'):
    """
    Resample flight state vector data to a specified time interval.
    
    This function takes raw flight state data with irregular time intervals and resamples it
    to create evenly spaced observations. For each new time interval, it takes the first 
    available state vector if multiple observations exist within that interval.
    
    Args:
        df (pd.DataFrame): DataFrame containing flight state vector data. Must have a 'time' 
                          column with Unix timestamps.
        interval (str): Pandas time interval string specifying the desired sampling rate.
                       Default is '5s' for 5 seconds. See pandas documentation for other options.
        
    Returns:
        pd.DataFrame: A new DataFrame with state vectors resampled to the specified interval.
                     The time column is converted to datetime and all rows are sorted chronologically.
    """
    # Create a copy to avoid modifying the original data
    df_copy = df.copy()
    
    # Convert Unix timestamps to pandas datetime objects if needed
    if df_copy['time'].dtype != 'datetime64[ns]':
        df_copy['time'] = pd.to_datetime(df_copy['time'], unit='s')
    
    # Ensure data is sorted chronologically
    df_copy = df_copy.sort_values('time')
    
    # Set time as index for resampling
    df_copy.set_index('time', inplace=True)
    
    # Store original column order
    original_columns = df_copy.columns.tolist()
    
    # Resample data and forward fill missing values
    resampled_df = df_copy.resample(interval).first()
    
    # Reset index to make time a regular column again
    resampled_df.reset_index(inplace=True)
    
    # Ensure all original columns are present in the same order
    for col in original_columns:
        if col not in resampled_df.columns:
            resampled_df[col] = None
            
    resampled_df = resampled_df[['time'] + original_columns]
    
    return resampled_df

def create_input_output_sequences(
    df_flight, 
    input_len=17, 
    pred_len=1, 
    feature_cols=None, 
    target_cols=None
):
    """
    Create pairs of (X, y) from a single flight's data for one-step-ahead prediction:
      - X: the first 'input_len' time steps
      - y: the next 'pred_len' time steps (1 in this case).
    """
    # If not specified, take all numeric columns except flight_id/time
    if feature_cols is None:
        feature_cols = [c for c in df_flight.columns if c not in ['flight_id', 'time']]
    if target_cols is None:
        target_cols = feature_cols

    feature_values = df_flight[feature_cols].values
    target_values = df_flight[target_cols].values

    X_list, y_list = [], []
    n = len(df_flight)
    seq_len = input_len + pred_len

    for start_idx in range(n - seq_len + 1):
        X_seq = feature_values[start_idx : start_idx + input_len]
        y_seq = target_values[start_idx + input_len : start_idx + input_len + pred_len]
        X_list.append(X_seq)
        y_list.append(y_seq)
        
    X_array = np.array(X_list)  # (num_samples, input_len, num_features)
    y_array = np.array(y_list)  # (num_samples, pred_len, num_targets)

    return X_array, y_array

def main():
    # 1. Load raw paths
    flight_paths = glob.glob(str(DATA_DIR / "raw/accident_flight_states/*"))
    print("[INFO] Found {} flight files.".format(len(flight_paths)))

    # 2. Resample each flight
    resampled_dfs = []
    for path in flight_paths:
        try:
            df = pd.read_parquet(path)
            # Resample
            df_res = resample_flight_state_data(df, interval='3s')

            # Drop columns that are entirely NaN
            df_res = df_res.dropna(axis=1, how='all').sort_values('time')

            # set index
            df_res.set_index('time', inplace=True)

            # Interpolate selected columns
            for col in ['lat','lon','velocity','heading','vertrate','geoaltitude','baroaltitude']:
                if col in df_res.columns:
                    df_res[col] = df_res[col].interpolate(method='time')
            
            df_res.dropna(subset=['lat','lon','velocity','heading','vertrate','geoaltitude','baroaltitude'],
                          inplace=True)
            
            # reset index
            df_res.reset_index(inplace=True)
            
            if not df_res.empty:
                resampled_dfs.append(df_res)
            
        except Exception as e:
            print(f"[WARNING] Could not process {path}: {e}")

    # Check if any valid DataFrames were processed
    if not resampled_dfs:
        print("[WARNING] No valid flight files were processed. Exiting.")
        return

    # 3. Combine into single DataFrame
    combined_df = pd.concat(resampled_dfs, ignore_index=True)
    print("[INFO] Combined DataFrame shape:", combined_df.shape)

    # 4. Create sliding windows per flight (group by 'icao24' or appropriate ID)
    all_X = []
    all_y = []
    feature_cols = ['lon','lat','heading','velocity','vertrate','heading','geoaltitude']
    target_cols  = ['lon','lat','heading','velocity','vertrate','heading','geoaltitude']

    # Adjust input_len if desired
    input_len = 29
    pred_len  = 1

    for icao24, df_group in combined_df.groupby('icao24'):
        df_group = df_group.sort_values('time').reset_index(drop=True)

        X_f, y_f = create_input_output_sequences(
            df_group,
            input_len=input_len,
            pred_len=pred_len,
            feature_cols=feature_cols,
            target_cols=target_cols
        )
        if len(X_f) > 0:
            all_X.append(X_f)
            all_y.append(y_f)

    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    print("[INFO] X_final shape:", X_final.shape)
    print("[INFO] y_final shape:", y_final.shape)

    # 5. Split data
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_final, y_final, test_size=0.2, shuffle=True, random_state=42
    )
    # 6. Scale
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Reshape to 2D
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    X_test_2d  = X_test_raw.reshape(-1,  X_test_raw.shape[-1])
    y_train_2d = y_train_raw.reshape(-1, y_train_raw.shape[-1])
    y_test_2d  = y_test_raw.reshape(-1,  y_test_raw.shape[-1])

    X_train_scaled = X_scaler.fit_transform(X_train_2d)
    X_test_scaled  = X_scaler.transform(X_test_2d)
    y_train_scaled = y_scaler.fit_transform(y_train_2d)
    y_test_scaled  = y_scaler.transform(y_test_2d)

    # Reshape back to 3D
    X_train = X_train_scaled.reshape(X_train_raw.shape)
    X_test  = X_test_scaled.reshape(X_test_raw.shape)
    y_train = y_train_scaled.reshape(y_train_raw.shape)
    y_test  = y_test_scaled.reshape(y_test_raw.shape)

    # 7. Save data + scalers
    os.makedirs('state_prediction/model/scalers', exist_ok=True)
    with open('state_prediction/model/scalers/X_scaler.pkl', 'wb') as f:
        pickle.dump(X_scaler, f)
    with open('state_prediction/model/scalers/y_scaler.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)

    os.makedirs('state_prediction/model/train_data', exist_ok=True)
    np.save('state_prediction/model/train_data/X_train.npy', X_train)
    np.save('state_prediction/model/train_data/y_train.npy', y_train)
    np.save('state_prediction/model/train_data/X_test.npy',  X_test)
    np.save('state_prediction/model/train_data/y_test.npy',  y_test)
    print("[INFO] Finished preparing data.")

if __name__ == "__main__":
    main()   