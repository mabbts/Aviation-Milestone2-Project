"""
Eastern US Flight Data Extractor
-------------------------------
Extracts and processes flight data from OpenSky Network's state_vectors_data4 table
for the Eastern United States region.
"""

from pyopensky.trino import Trino
from datetime import datetime, timedelta
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Eastern US Boundaries
EAST_US_BOUNDS = {
    'north': 49.00,    # Northern Maine
    'south': 24.50,    # Southern Florida
    'west': -105.00,   # Mississippi River approx
    'east': -66.90     # Eastern Maine
}

def get_east_us_data(start_time: datetime, 
                     duration_hours: int = 1,
                     cached: bool = True) -> pd.DataFrame:
    """
    Retrieve state vectors data for Eastern US airspace.
    
    Args:
        start_time: Start time for the query (UTC)
        duration_hours: How many hours of data to retrieve
        cached: Whether to use cached data
    
    Returns:
        DataFrame containing state vectors data
    """
    
    trino = Trino()
    
    bounds = (
        EAST_US_BOUNDS['west'],
        EAST_US_BOUNDS['south'],
        EAST_US_BOUNDS['east'],
        EAST_US_BOUNDS['north']
    )
    
    try:
        df = trino.history(
            start=start_time,
            stop=start_time + timedelta(hours=duration_hours),
            bounds=bounds,
            # Updated column names to match schema
            selected_columns=(
                "time",
                "icao24",
                "callsign",
                "lat",           # Changed from latitude
                "lon",           # Changed from longitude
                "geoaltitude",   # Changed from altitude
                "velocity",
                "heading",
                "vertrate",      # Changed from vertical_rate
                "onground"
            ),
            cached=cached,
            compress=True
        )
        
        if df is not None:
            # Updated column names in the query
            df = df.query(
                "lat >= @EAST_US_BOUNDS['south'] and "  # Changed from latitude
                "lat <= @EAST_US_BOUNDS['north'] and "  # Changed from latitude
                "lon >= @EAST_US_BOUNDS['west'] and "   # Changed from longitude
                "lon <= @EAST_US_BOUNDS['east']"        # Changed from longitude
            )
            
            # Optionally rename columns for clarity
            df = df.rename(columns={
                'lat': 'latitude',
                'lon': 'longitude',
                'geoaltitude': 'altitude'
            })
            
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        raise

def process_large_timeframe(start_time: datetime, 
                          total_hours: int = 24,
                          chunk_hours: int = 1):
    """
    Process larger time periods by splitting into smaller chunks.
    
    Args:
        start_time: Starting time for data collection
        total_hours: Total number of hours to process
        chunk_hours: Size of each chunk in hours
    """
    
    all_data = []
    
    for hour in range(0, total_hours, chunk_hours):
        chunk_start = start_time + timedelta(hours=hour)
        logger.info(f"Processing chunk starting at {chunk_start}")
        
        df = get_east_us_data(
            start_time=chunk_start,
            duration_hours=chunk_hours
        )
        
        if not df.empty:
            # Save each chunk to avoid memory issues
            chunk_file = f"east_us_data_{chunk_start.strftime('%Y%m%d_%H')}.parquet"
            df.to_parquet(chunk_file)
            logger.info(f"Saved chunk to {chunk_file}")
            
            all_data.append(df)
    
    if all_data:
        # Combine all chunks
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_parquet("east_us_complete_dataset.parquet")
        logger.info("Saved complete dataset")
        return final_df
    
    return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    now = datetime.utcnow()
    start_time = now - timedelta(hours=24)  # Get last 24 hours
    
    # Process data in 1-hour chunks
    df = process_large_timeframe(
        start_time=start_time,
        total_hours=24,
        chunk_hours=1
    )
    
    if not df.empty:
        print(f"Total flights processed: {len(df)}")
        print("\nSample of data:")
        print(df.head())
