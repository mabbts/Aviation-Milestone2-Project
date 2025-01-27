"""
OpenSky Data Extractor
---------------------
Extracts flight data from OpenSky Network tables focusing on features relevant
for flight pattern analysis and clustering.

Tables used:
- state_vectors_data4: Real-time aircraft state vectors
- flights_data4: Flight path and airport information
- position_data4: Detailed position information
"""

from pyopensky.trino import Trino
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import Tuple, Optional

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

def get_state_vectors(
    trino: Trino,
    start_time: datetime,
    end_time: datetime,
    bounds: Tuple[float, float, float, float],
    cached: bool = True
) -> pd.DataFrame:
    """
    Extract state vectors data containing core flight tracking information.
    
    Uses state_vectors_data4 table which contains:
    - time: integer (timestamp)
    - icao24: varchar (unique aircraft identifier)
    - lat, lon: double (position)
    - velocity: double (ground speed)
    - heading: double (true track)
    - vertrate: double (vertical rate)
    - callsign: varchar (flight number/identifier)
    - geoaltitude: double (geometric altitude)
    """
    try:
        return trino.history(
            start=start_time,
            stop=end_time,
            bounds=bounds,
            selected_columns=(
                "time", "icao24", "callsign", "lat", "lon",
                "velocity", "heading", "vertrate", "geoaltitude",
                "onground", "lastposupdate", "lastcontact"
            ),
            cached=cached,
            compress=True
        )
    except Exception as e:
        logger.error(f"Error retrieving state vectors: {str(e)}")
        return pd.DataFrame()

def get_flight_paths(
    trino: Trino,
    start_time: datetime,
    cached: bool = True
) -> pd.DataFrame:
    """
    Extract flight path data containing origin/destination information.
    
    Uses flights_data4 table which contains:
    - icao24: varchar (aircraft identifier)
    - estdepartureairport: varchar
    - estarrivalairport: varchar
    - track: array of position records
    """
    try:
        # Convert datetime to day integer for partition key
        day = int(start_time.timestamp() // 86400)
        
        query = f"""
        SELECT icao24, firstseen, estdepartureairport, 
               lastseen, estarrivalairport, callsign
        FROM flights_data4
        WHERE day = {day}
        """
        
        return trino.query(query, cached=cached)
    except Exception as e:
        logger.error(f"Error retrieving flight paths: {str(e)}")
        return pd.DataFrame()

def get_position_details(
    trino: Trino,
    start_time: datetime,
    cached: bool = True
) -> pd.DataFrame:
    """
    Extract detailed position data with additional metrics.
    
    Uses position_data4 table which contains:
    - icao24: varchar (aircraft identifier)
    - lat, lon: double (position)
    - groundspeed: double
    - heading: double
    - baroalt: boolean (barometric altitude available)
    """
    try:
        # Convert datetime to hour integer for partition key
        hour = int(start_time.timestamp() // 3600)
        
        query = f"""
        SELECT icao24, mintime, maxtime, lat, lon, 
               groundspeed, heading, baroalt, alt
        FROM position_data4
        WHERE hour = {hour}
        """
        
        return trino.query(query, cached=cached)
    except Exception as e:
        logger.error(f"Error retrieving position details: {str(e)}")
        return pd.DataFrame()

def extract_georgia_flights(
    start_time: datetime,
    duration_hours: int = 1,
    cached: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract flight data for Georgia airspace from multiple OpenSky tables.
    
    Args:
        start_time: Start time for the query (UTC)
        duration_hours: How many hours of data to retrieve
        cached: Whether to use cached data
    
    Returns:
        Tuple of DataFrames (state_vectors, flight_paths, position_details)
    """
    trino = Trino()
    end_time = start_time + timedelta(hours=duration_hours)
    
    bounds = (
        GEORGIA_BOUNDS['west'],
        GEORGIA_BOUNDS['south'],
        GEORGIA_BOUNDS['east'],
        GEORGIA_BOUNDS['north']
    )
    
    # Extract data from each table
    state_vectors = get_state_vectors(trino, start_time, end_time, bounds, cached)
    flight_paths = get_flight_paths(trino, start_time, cached)
    position_details = get_position_details(trino, start_time, cached)
    
    # Log extraction summary
    logger.info(f"Extracted {len(state_vectors)} state vector records")
    logger.info(f"Extracted {len(flight_paths)} flight path records")
    logger.info(f"Extracted {len(position_details)} position detail records")
    
    return state_vectors, flight_paths, position_details

if __name__ == "__main__":
    # Example usage
    now = datetime.utcnow()
    start_time = now - timedelta(hours=24)
    
    vectors, paths, positions = extract_georgia_flights(
        start_time=start_time,
        duration_hours=24
    )
    
    # Save raw extracted data
    if not vectors.empty:
        vectors.to_parquet("/home/jhovan/GitHub/Aviation-Milestone2-Project/data/raw_state_vectors.parquet") 
    if not paths.empty:
        paths.to_parquet("/home/jhovan/GitHub/Aviation-Milestone2-Project/data/raw_flight_paths.parquet")
    if not positions.empty:
        positions.to_parquet("/home/jhovan/GitHub/Aviation-Milestone2-Project/data/raw_position_details.parquet")
