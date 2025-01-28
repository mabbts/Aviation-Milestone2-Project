from pyopensky.trino import Trino
from datetime import datetime
import pandas as pd
import time

# Convert datetime string to Unix timestamp for a specific hour
sample_start = int(datetime.strptime('2025-01-14 12:00:00', '%Y-%m-%d %H:%M:%S').timestamp())
sample_end = int(datetime.strptime('2025-01-14 13:00:00', '%Y-%m-%d %H:%M:%S').timestamp())

# Initialize Trino connection
trino = Trino()

# Dictionary of queries for each table
queries = {
    'flights_data5': """
        SELECT *
        FROM flights_data5
        WHERE firstseen BETWEEN {start_time} AND {end_time}
        LIMIT 100
    """,
    
    'state_vectors_data4': """
        SELECT *
        FROM state_vectors_data4
        WHERE time BETWEEN {start_time} AND {end_time}
        LIMIT 100
    """,
    
    'position_data4': """
        SELECT *
        FROM position_data4
        WHERE mintime BETWEEN {start_time} AND {end_time}
        LIMIT 100
    """,
    
    'velocity_data4': """
        SELECT *
        FROM velocity_data4
        WHERE mintime BETWEEN {start_time} AND {end_time}
        LIMIT 100
    """,
    
    'identification_data4': """
        SELECT *
        FROM identification_data4
        WHERE mintime BETWEEN {start_time} AND {end_time}
        LIMIT 100
    """,
    
    'operational_status_data4': """
        SELECT *
        FROM operational_status_data4
        WHERE mintime BETWEEN {start_time} AND {end_time}
        LIMIT 100
    """,
    
    'rollcall_replies_data4': """
        SELECT *
        FROM rollcall_replies_data4
        WHERE mintime BETWEEN {start_time} AND {end_time}
        LIMIT 100
    """,
    
    'flarm_raw': """
        SELECT *
        FROM flarm_raw
        WHERE timestamp BETWEEN {start_time} AND {end_time}
        LIMIT 100
    """
}

# Download and save samples
for table_name, query in queries.items():
    print(f"Downloading sample from {table_name}...")
    
    try:
        # Format query with timestamps
        formatted_query = query.format(
            start_time=sample_start,
            end_time=sample_end
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