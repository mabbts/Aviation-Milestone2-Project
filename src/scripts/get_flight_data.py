from ..database.queries import OpenSkyQueries
from pyopensky.trino import Trino
import pandas as pd
from datetime import datetime, timedelta
from ..utils.paths import DATA_DIR
trino = Trino()

def get_detailed_flights(start_time, end_time):
    # Convert to UNIX timestamps in seconds
    start_unix = int(start_time.timestamp())
    end_unix = int(end_time.timestamp())
    
    # Get and execute the query
    query = OpenSkyQueries.get_flight_data_v4(start_unix, end_unix)
    result = trino.query(query)
    
    # Convert to DataFrame
    return pd.DataFrame(result)

def main():
    # Define PRECISE time range (1 hour window)
    start_time = datetime(2025, 1, 1, 1, 0)  # Jan 1, 2025 1:00 UTC
    end_time = start_time + timedelta(hours=1)
    
    # Get the aggregated flight data
    df = get_detailed_flights(start_time, end_time)
    
    # Display the results
    print(f"Flight details from {start_time} to {end_time}:")
    print(df)
    
    # Save to CSV with timestamp in filename
    filename = f"flight_details_{start_time:%Y%m%d_%H%M}.csv"
    df.to_csv(DATA_DIR / filename, index=False)

if __name__ == "__main__":
    main()