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

query_aggregate = """
SELECT
    f.icao24,
    f.callsign,
    f.airportofdeparture AS src,
    f.airportofdestination AS dest,
    AVG(s.velocity) AS avg_velocity,
    AVG(s.geoaltitude) AS avg_geo_altitude,
    MAX(s.geoaltitude) AS max_geo_altitude,
    MIN(s.geoaltitude) AS min_geo_altitude,
    COUNT(s.time) AS num_state_updates
FROM
    flights_data5 f
JOIN
    state_vectors_data4 s
ON
    f.icao24 = s.icao24
    AND f.callsign = s.callsign
WHERE
    s.time BETWEEN {start_time} AND {end_time}
    AND s.lat BETWEEN {south} AND {north}
    AND s.lon BETWEEN {west} AND {east}
GROUP BY
    f.icao24, f.callsign, f.airportofdeparture, f.airportofdestination
"""

aggregate_df = trino.query(query_aggregate.format(
    start_time=start_time,
    end_time=end_time,
    north=GEORGIA_BOUNDS['north'],
    south=GEORGIA_BOUNDS['south'],
    east=GEORGIA_BOUNDS['east'],
    west=GEORGIA_BOUNDS['west']
))

aggregate_df.to_csv('aggregate_df.csv', index=False)