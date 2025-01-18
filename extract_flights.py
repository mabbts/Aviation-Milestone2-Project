"""
Georgia Flight Data Extractor with Aircraft Information
-------------------------------------------------------
Extracts, processes, and classifies flight data for the Georgia region,
incorporating aircraft type and flight length classification.
"""

from pyopensky.trino import Trino
from datetime import datetime, timedelta
import pandas as pd
import logging
from haversine import haversine, Unit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Georgia State bounds
GEORGIA_BOUNDS = {
    'north': 35.00,  # max latitude
    'south': 30.63,  # min latitude
    'west': -85.13,  # min longitude
    'east': -80.85   # max longitude
}

def get_georgia_data(start_time: datetime, duration_hours: int = 1, cached: bool = True) -> pd.DataFrame:
    """
    Retrieve state vectors data for Georgia airspace, including aircraft info.
    
    Args:
        start_time: Start time for the query (UTC)
        duration_hours: How many hours of data to retrieve
        cached: Whether to use cached data
    
    Returns:
        DataFrame containing state vectors data with additional aircraft details
    """
    trino = Trino()

    bounds = (
        GEORGIA_BOUNDS['west'],
        GEORGIA_BOUNDS['south'],
        GEORGIA_BOUNDS['east'],
        GEORGIA_BOUNDS['north']
    )

    try:
        # Retrieve flight data
        flight_data = trino.history(
            start=start_time,
            stop=start_time + timedelta(hours=duration_hours),
            bounds=bounds,
            selected_columns=(
                "time", "icao24", "callsign", "lat", "lon",
                "geoaltitude", "velocity", "heading", "vertrate", "onground"
            ),
            cached=cached,
            compress=True
        )
        
        if flight_data is not None:
            # Filter to Georgia bounds
            flight_data = flight_data.query(
                "lat >= @GEORGIA_BOUNDS['south'] and " 
                "lat <= @GEORGIA_BOUNDS['north'] and " 
                "lon >= @GEORGIA_BOUNDS['west'] and "
                "lon <= @GEORGIA_BOUNDS['east']"
            )
            
            # Rename columns for clarity
            flight_data.rename(columns={'lat': 'latitude', 'lon': 'longitude', 'geoaltitude': 'altitude'}, inplace=True)
            
            # Add aircraft type information
            aircraft_info = trino.query(
                "SELECT icao24, typecode, manufacturer FROM aircraft_database"
            )
            flight_data = flight_data.merge(aircraft_info, on='icao24', how='left')
            
            return flight_data
        
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        raise

def calculate_flight_metrics(flight_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate flight duration and distance, categorize flights by length.
    
    Args:
        flight_data: DataFrame containing raw flight data
    
    Returns:
        DataFrame with additional metrics
    """
    if 'time' not in flight_data.columns or flight_data.empty:
        logger.warning("Flight data is empty or missing required columns.")
        return flight_data

    # Sort and calculate flight duration
    flight_data.sort_values(by=['icao24', 'time'], inplace=True)
    flight_data['flight_duration'] = flight_data.groupby('icao24')['time'].diff().dt.total_seconds() / 3600

    # Calculate distance between points
    def compute_distance(row):
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            return haversine(
                (row['latitude'], row['longitude']),
                (row['latitude_prev'], row['longitude_prev']),
                unit=Unit.NAUTICAL_MILES
            )
        return 0

    flight_data['latitude_prev'] = flight_data['latitude'].shift()
    flight_data['longitude_prev'] = flight_data['longitude'].shift()
    flight_data['distance_nm'] = flight_data.apply(compute_distance, axis=1)

    # Categorize flight lengths
    bins = [0, 50, 300, float('inf')]
    labels = ['local', 'regional', 'cross-country']
    flight_data['flight_category'] = pd.cut(flight_data['distance_nm'], bins=bins, labels=labels, right=False)

    return flight_data.drop(['latitude_prev', 'longitude_prev'], axis=1)

if __name__ == "__main__":
    # Example usage
    now = datetime.utcnow()
    start_time = now - timedelta(hours=12)

    # Get raw flight data
    raw_data = get_georgia_data(start_time=start_time, duration_hours=12)

    # Process and analyze flight data
    if not raw_data.empty:
        processed_data = calculate_flight_metrics(raw_data)
        processed_data.to_parquet("data/georgia_flight_metrics.parquet")
        logger.info("Saved processed flight data with metrics.")
        print(processed_data.head())
