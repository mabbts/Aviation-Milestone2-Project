from pyopensky.trino import Trino
from datetime import datetime
import time

start_time = int(datetime.strptime('2025-01-14', '%Y-%m-%d').timestamp())
end_time = int(datetime.strptime('2025-01-15', '%Y-%m-%d').timestamp())

# Georgia bounds
GEORGIA_BOUNDS = {
    'north': 35.00,  # max latitude
    'south': 30.63,  # min latitude
    'west': -85.13,  # min longitude
    'east': -80.85   # max longitude
}

trino = Trino()

query_waypoint = """
SELECT
    icao24,
    firstseen,
    lastseen, 
    callsign,
    t.time AS track_time,
    t.latitude AS track_lat,
    t.longitude AS track_lon,
    t.altitude AS track_alt
FROM flights_data4
CROSS JOIN UNNEST(track) AS t
WHERE
    t.time BETWEEN {start_time} AND {end_time}
    AND t.latitude BETWEEN {south} AND {north}
    AND t.longitude BETWEEN {west} AND {east}
"""

waypoint_df = trino.query(query_waypoint.format(
    start_time=start_time,
    end_time=end_time,
    north=GEORGIA_BOUNDS['north'],
    south=GEORGIA_BOUNDS['south'],
    east=GEORGIA_BOUNDS['east'],
    west=GEORGIA_BOUNDS['west']
))
waypoint_df.to_csv('waypoint_df.csv', index=False)