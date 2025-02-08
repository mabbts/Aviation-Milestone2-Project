"""
This module provides functions for preprocessing flight data.
It includes a function to compute aggregate metrics from the 'track' column.
"""

import numpy as np
import pandas as pd

def compute_track_metrics(track):
    """
    Given a track as a NumPy-like array of shape (N, 6) or (N,),
    where each inner element is [time, latitude, longitude, altitude, heading, onground],
    compute various aggregate metrics (min, max, mean) for each dimension,
    plus the fraction of time on-ground, and convert the minimum and maximum times
    to human-readable datetime.
    
    Returns:
    - dict: A dictionary with the following keys:
        time_min, time_max, time_avg, start_time, end_time,
        latitude_min, latitude_max, latitude_avg,
        longitude_min, longitude_max, longitude_avg,
        altitude_min, altitude_max, altitude_avg,
        heading_min, heading_max, heading_avg,
        onground_percentage
    """
    # If the track is None or empty, return None.
    if track is None or len(track) == 0:
        return None

    # Convert the track to a NumPy array (in case it's a list)
    track = np.array(track, dtype=object)
    
    # If the array is 1D and each element is a list/array of 6 elements,
    # stack them into a 2D array.
    if len(track.shape) == 1 and isinstance(track[0], (list, np.ndarray)):
        data = np.stack(track, axis=0)
    else:
        data = track

    # At this point, data should be a NumPy array of shape (N, 6)
    # Separate the columns by casting to float as needed.
    times     = data[:, 0].astype(float)
    latitudes = data[:, 1].astype(float)
    longitudes= data[:, 2].astype(float)
    altitudes = data[:, 3].astype(float)
    headings  = data[:, 4].astype(float)

    # For the onground indicator, try converting to int and then calculate the percentage.
    try:
        onground = data[:, 5].astype(int)
        onground_frac = onground.mean() * 100.0
    except Exception:
        onground_frac = None

    metrics = {
        # Time-based metrics
        "time_min":      times.min(),
        "time_max":      times.max(),
        "time_avg":      times.mean(),
        "start_time":    pd.to_datetime(times.min(), unit="s"),
        "end_time":      pd.to_datetime(times.max(), unit="s"),
        # Latitude metrics
        "latitude_min":  latitudes.min(),
        "latitude_max":  latitudes.max(),
        "latitude_avg":  latitudes.mean(),
        # Longitude metrics
        "longitude_min": longitudes.min(),
        "longitude_max": longitudes.max(),
        "longitude_avg": longitudes.mean(),
        # Altitude metrics
        "altitude_min":  altitudes.min(),
        "altitude_max":  altitudes.max(),
        "altitude_avg":  altitudes.mean(),
        # Heading metrics
        "heading_min":   headings.min(),
        "heading_max":   headings.max(),
        "heading_avg":   headings.mean(),
        # On-ground fraction (percentage)
        "onground_percentage": onground_frac,
    }
    return metrics

def preprocess_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the flight DataFrame by computing aggregate metrics
    for the 'track' column and appending these metrics as new columns.
    
    Parameters:
    - df (pd.DataFrame): Flight data that must include a 'track' column.
    
    Returns:
    - pd.DataFrame: The flight DataFrame with the computed aggregate metric columns.
    """
    if "track" in df.columns:
        # compute metrics, wrapping the result into a pd.Series so that each metric becomes a column.
        df_metrics = df["track"].apply(lambda x: pd.Series(compute_track_metrics(x)))
        # Concatenate the new metric columns with the original DataFrame.
        df = pd.concat([df, df_metrics], axis=1)
    else:
        print("Warning: 'track' column not found in the input DataFrame.")
    
    return df

