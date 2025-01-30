from typing import Dict, Optional
from datetime import datetime

class OpenSkyQueries:
    """Centralized storage for all OpenSky SQL queries"""
    
    @staticmethod
    def get_flight_aggregate(
        start_time: int,
        end_time: int,
        bounds: Dict[str, float]
    ) -> str:
        """Get aggregated flight statistics within geographical bounds"""
        return """
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
        """.format(
            start_time=start_time,
            end_time=end_time,
            **bounds
        )

    @staticmethod
    def get_flight_path(
        start_time: int,
        end_time: int,
        icao24: Optional[str] = None
    ) -> str:
        """Get detailed flight path data"""
        where_clause = f"AND icao24 = '{icao24}'" if icao24 else ""
        
        return """
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
            {where_clause}
        ORDER BY firstseen
        """.format(
            start_time=start_time,
            end_time=end_time,
            where_clause=where_clause
        )
