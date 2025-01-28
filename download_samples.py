from pyopensky.trino import Trino
from datetime import datetime
import pandas as pd
import time
import os

# Georgia bounds
GEORGIA_BOUNDS = {
    'north': 35.00,  # max latitude
    'south': 30.63,  # min latitude
    'west': -85.13,  # min longitude
    'east': -80.85   # max longitude
}

# Convert datetime string to Unix timestamp for a specific hour
sample_start = int(datetime.strptime('2024-01-14 12:00:00', '%Y-%m-%d %H:%M:%S').timestamp())
sample_end = int(datetime.strptime('2024-01-14 13:00:00', '%Y-%m-%d %H:%M:%S').timestamp())

# Initialize Trino connection
trino = Trino()

# ensure samples folder exists
if not os.path.exists('samples'):
    os.makedirs('samples')

# Dictionary of queries for each table
queries = {
    'flights_data5': """
        SELECT *
        FROM flights_data5
        WHERE firstseen BETWEEN {start_time} AND {end_time}
        AND array_any(t -> t.latitude BETWEEN {south} AND {north}
            AND t.longitude BETWEEN {west} AND {east}, track)
        LIMIT 100
    """,
    
    'state_vectors_data4': """
        SELECT *
        FROM state_vectors_data4
        WHERE time BETWEEN {start_time} AND {end_time}
        AND lat BETWEEN {south} AND {north}
        AND lon BETWEEN {west} AND {east}
        LIMIT 100
    """,
    
    'position_data4': """
        SELECT *
        FROM position_data4
        WHERE mintime BETWEEN {start_time} AND {end_time}
        AND lat BETWEEN {south} AND {north}
        AND lon BETWEEN {west} AND {east}
        LIMIT 100
    """,
    
    'velocity_data4': """
        SELECT *
        FROM velocity_data4 v
        JOIN position_data4 p
        ON v.icao24 = p.icao24 AND v.mintime = p.mintime
        WHERE v.mintime BETWEEN {start_time} AND {end_time}
        AND p.lat BETWEEN {south} AND {north}
        AND p.lon BETWEEN {west} AND {east}
        LIMIT 100
    """,
    
    'identification_data4': """
        SELECT i.*
        FROM identification_data4 i
        JOIN position_data4 p
        ON i.icao24 = p.icao24 AND i.mintime = p.mintime
        WHERE i.mintime BETWEEN {start_time} AND {end_time}
        AND p.lat BETWEEN {south} AND {north}
        AND p.lon BETWEEN {west} AND {east}
        LIMIT 100
    """,
    
    'operational_status_data4': """
        SELECT o.*
        FROM operational_status_data4 o
        JOIN position_data4 p
        ON o.icao24 = p.icao24 AND o.mintime = p.mintime
        WHERE o.mintime BETWEEN {start_time} AND {end_time}
        AND p.lat BETWEEN {south} AND {north}
        AND p.lon BETWEEN {west} AND {east}
        LIMIT 100
    """,
    
    'rollcall_replies_data4': """
        SELECT r.*
        FROM rollcall_replies_data4 r
        JOIN position_data4 p
        ON r.icao24 = p.icao24 AND r.mintime = p.mintime
        WHERE r.mintime BETWEEN {start_time} AND {end_time}
        AND p.lat BETWEEN {south} AND {north}
        AND p.lon BETWEEN {west} AND {east}
        LIMIT 100
    """,
    
    'flarm_raw': """
        SELECT *
        FROM flarm_raw
        WHERE timestamp BETWEEN {start_time} AND {end_time}
        AND sensorlatitude BETWEEN {south} AND {north}
        AND sensorlongitude BETWEEN {west} AND {east}
        LIMIT 100
    """
}

# Download and save samples
for table_name, query in queries.items():
    print(f"Downloading sample from {table_name}...")
    
    try:
        # Format query with timestamps and bounds
        formatted_query = query.format(
            start_time=sample_start,
            end_time=sample_end,
            north=GEORGIA_BOUNDS['north'],
            south=GEORGIA_BOUNDS['south'],
            east=GEORGIA_BOUNDS['east'],
            west=GEORGIA_BOUNDS['west']
        )
        
        # Execute query
        df = trino.query(formatted_query)
        
        # Save to CSV
        output_file = f"samples/{table_name}_sample.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved sample to {output_file}")
        
        # Add a small delay between queries
        time.sleep(1)
        
    except Exception as e:
        print(f"Error downloading {table_name}: {str(e)}")

print("Sample collection complete!") 