from pyopensky.trino import Trino
from datetime import datetime
import time

# Convert datetime strings to Unix timestamps
start_time = int(datetime.strptime('2024-03-01 08:00:00', '%Y-%m-%d %H:%M:%S').timestamp())
end_time = int(datetime.strptime('2024-03-01 09:00:00', '%Y-%m-%d %H:%M:%S').timestamp())

trino = Trino()

df = trino.query(f"""
SELECT
  sv.time,
  sv.icao24,
  sv.lat,
  sv.lon,
  sv.velocity,
  sv.baroaltitude,
  sv.callsign
FROM state_vectors_data4 sv
JOIN (
  SELECT 
    f.icao24,
    f.callsign,
    f.firstseen,
    f.lastseen
  FROM flights_data4 f
  WHERE f.firstseen BETWEEN {start_time} AND {end_time}
    AND (f.estdepartureairport = 'KATL' OR f.estarrivalairport = 'KATL')
) AS flights
ON sv.icao24 = flights.icao24 AND sv.callsign = flights.callsign
WHERE
  sv.time BETWEEN {start_time} AND {end_time}
  AND sv.lon BETWEEN -85.13 AND -80.85
  AND sv.lat BETWEEN 30.63 AND 35.0
""")

print(df)