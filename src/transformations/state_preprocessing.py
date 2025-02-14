import pandas as pd

def resample_flight_state_data(df, interval='5s'):
    """
    Resample flight data at specified time intervals.
    
    Args:
        df: DataFrame containing flight data with a 'time' column
        interval: String specifying the resampling interval (default '5S' for 5 seconds)
        
    Returns:
        Resampled DataFrame
    """
    # Work on a copy of the dataframe
    df_copy = df.copy()
    
    # Convert the 'time' column to datetime
    df_copy['time'] = pd.to_datetime(df_copy['time'], unit='s')
    
    # Sort by time
    df_copy = df_copy.sort_values('time')
    
    # Set time as index
    df_copy.set_index('time', inplace=True)
    
    # Resample and take first record in each interval
    resampled_df = df_copy.resample(interval).first().reset_index()
    
    return resampled_df
