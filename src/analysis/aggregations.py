from datetime import datetime
from pyopensky.trino import Trino
import pandas as pd
from ..database.queries import OpenSkyQueries
from ..utils.constants import GEORGIA_BOUNDS

def get_flight_aggregates(start_date: str, end_date: str) -> pd.DataFrame:
    """Get aggregated flight data for a date range"""
    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_time = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    
    trino = Trino()
    query = OpenSkyQueries.get_flight_aggregate(
        start_time=start_time,
        end_time=end_time,
        bounds=GEORGIA_BOUNDS
    )
    
    return trino.query(query) 

def get_detailed_flights(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get detailed flight data including tracks and durations for a date range.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame containing flight details including duration and track data
    """
    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_time = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    
    trino = Trino()
    query = OpenSkyQueries.get_detailed_flight_data(
        start_time=start_time,
        end_time=end_time,
    )
    
    return trino.query(query) 

