import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def convert_to_ecef(lat, lon, alt):
    """
    Convert latitude, longitude, altitude to ECEF coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters (above sea level)
        
    Returns:
        Tuple of (X, Y, Z) coordinates in meters
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis (meters)
    f = 1/298.257223563  # flattening
    b = a * (1 - f)  # semi-minor axis
    e2 = 1 - (b**2)/(a**2)  # eccentricity squared
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Calculate N (radius of curvature)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    # Calculate ECEF coordinates
    X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    
    return X, Y, Z

def encode_heading(heading):
    """
    Encode heading using sin/cos transformation.
    
    Args:
        heading: Array of heading values in degrees
        
    Returns:
        Tuple of (sin_encoded, cos_encoded) values
    """
    heading_rad = np.radians(heading)
    return np.sin(heading_rad), np.cos(heading_rad)

def create_input_output_sequences(
    df_flight, 
    input_len=17, 
    pred_len=1, 
    feature_cols=None, 
    target_cols=None
):
    """
    Create sequences with ECEF coordinate transformations applied.
    """
    # Convert coordinates to ECEF
    X, Y, Z = convert_to_ecef(
        df_flight['lat'].values,
        df_flight['lon'].values,
        df_flight['geoaltitude'].values
    )
    
    # Encode heading
    heading_sin, heading_cos = encode_heading(df_flight['heading'].values)
    
    # Create transformed feature matrix
    transformed_features = np.column_stack([
        X, Y, Z,  # ECEF coordinates
        heading_sin, heading_cos,  # Encoded heading
        df_flight['velocity'].values,
        df_flight['vertrate'].values,
    ])
    
    # Create sequences
    X_list, y_list = [], []
    n = len(df_flight)
    seq_len = input_len + pred_len
    
    for start_idx in range(n - seq_len + 1):
        X_seq = transformed_features[start_idx : start_idx + input_len]
        y_seq = transformed_features[start_idx + input_len : start_idx + input_len + pred_len]
        
        X_list.append(X_seq)
        y_list.append(y_seq)
    
    X_array = np.array(X_list)
    y_array = np.array(y_list)
    
    return X_array, y_array

def ecef_to_lat_lon_alt(X, Y, Z):
    """
    Convert ECEF coordinates back to latitude, longitude, altitude.
    
    Args:
        X, Y, Z: ECEF coordinates in meters
        
    Returns:
        Tuple of (latitude, longitude, altitude) in degrees and meters
    """
    # WGS84 parameters
    a = 6378137.0  # semi-major axis
    f = 1/298.257223563  # flattening
    b = a * (1 - f)  # semi-minor axis
    e2 = 1 - (b**2)/(a**2)  # eccentricity squared
    
    # Calculate longitude
    lon = np.arctan2(Y, X)
    
    # Calculate latitude iteratively
    p = np.sqrt(X**2 + Y**2)
    lat = np.arctan2(Z, p * (1 - e2))
    
    for _ in range(5):  # Usually converges in 2-3 iterations
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(Z, p * (1 - e2 * N/(N + h)))
    
    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    
    return lat, lon, h


# Function to plot predictions
def plot_trajectory_prediction(model, input_seq, actual=None):
    """
    Plot predicted trajectory on a map.
    """
    with torch.no_grad():
        pred = model(input_seq)
        pred_np = pred.cpu().numpy()
        
        # Convert ECEF predictions back to lat/lon
        pred_lat, pred_lon, pred_alt = ecef_to_lat_lon_alt(
            pred_np[:, 0],  # X
            pred_np[:, 1],  # Y
            pred_np[:, 2]   # Z
        )
        
        # If actual trajectory is provided, convert it too
        if actual is not None:
            actual_np = actual.cpu().numpy()
            actual_lat, actual_lon, actual_alt = ecef_to_lat_lon_alt(
                actual_np[:, 0],
                actual_np[:, 1],
                actual_np[:, 2]
            )
        
        plt.figure(figsize=(12, 8))
        plt.plot(pred_lon, pred_lat, 'r.-', label='Predictions')
        if actual is not None:
            plt.plot(actual_lon, actual_lat, 'b.-', label='Actual')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Flight Trajectory Prediction')
        plt.legend()
        plt.grid(True)
        plt.show()