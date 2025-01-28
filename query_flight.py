from pyopensky.trino import Trino
from datetime import datetime
import time

# Convert datetime strings to Unix timestamps
start_time = int(datetime.strptime('2025-01-14', '%Y-%m-%d').timestamp())
end_time = int(datetime.strptime('2025-01-15', '%Y-%m-%d').timestamp())

# Specify the aircraft's icao24 code
icao24 = 'a4080c'  # Replace with your desired icao24 code

trino = Trino()

df = trino.query(f"""
SELECT
    firstseen,
    lastseen,
    callsign,
    estdepartureairport,
    estarrivalairport,
    takeofflatitude as departure_lat,
    takeofflongitude as departure_lon, 
    landinglatitude as arrival_lat,
    landinglongitude as arrival_lon
FROM flights_data5
WHERE
    firstseen BETWEEN {start_time} AND {end_time}
    AND icao24 = '{icao24}'
ORDER BY firstseen
""")

print(df)