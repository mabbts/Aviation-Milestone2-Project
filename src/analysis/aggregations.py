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