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

    @staticmethod
    def get_aircraft_flights(
        start_time: int,
        end_time: int,
        icao24: str
    ) -> str:
        """
        Get flight data for a specific aircraft within a time range.
        
        Args:
            start_time: Unix timestamp for start time
            end_time: Unix timestamp for end time
            icao24: Aircraft's ICAO24 identifier
            
        Returns:
            SQL query string
        """
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
            AND icao24 = '{icao24}'
        ORDER BY firstseen
        """.format(
            start_time=start_time,
            end_time=end_time,
            icao24=icao24
        )

    @staticmethod
    def get_detailed_flight_data(
        start_time: int,
        end_time: int,
    ) -> str:
        """
        Get detailed flight data including duration and track information within geographical bounds.
        
        Args:
            start_time: Unix timestamp for start time
            end_time: Unix timestamp for end time
            
        Returns:
            SQL query string that returns flight details including:
            - Basic flight identifiers (icao24, callsign)
            - Departure and arrival information
            - Flight timing and duration
            - Track data array
        """
        return """
        SELECT
            f.icao24,
            f.callsign,
            f.airportofdeparture,
            f.airportofdestination,
            f.takeofftime,
            f.landingtime,
            (f.landingtime - f.takeofftime) as flight_duration,
            f.takeofflatitude,
            f.takeofflongitude,
            f.landinglatitude,
            f.landinglongitude,
            f.track
        FROM
            flights_data5 f
        WHERE
            f.day BETWEEN FLOOR({start_time}/86400.0) AND FLOOR({end_time}/86400.0)
            AND f.takeofftime BETWEEN {start_time} AND {end_time}
        ORDER BY
            f.takeofftime
        """.format(
            start_time=start_time,
            end_time=end_time,
        )

    @staticmethod
    def get_flight_data_v4(
        start_time: int,
        end_time: int,
    ) -> str:
        """
        Get flight data from flights_data4 table with estimated airports and track information.
        
        Args:
            start_time: Unix timestamp for start time
            end_time: Unix timestamp for end time
            
        Returns:
            SQL query string that returns:
            - ICAO24 identifier
            - Callsign
            - Estimated departure/arrival airports
            - Flight timing
            - Track coordinates
        """
        return """
        SELECT
            f.icao24,
            f.callsign,
            f.estdepartureairport,
            f.estarrivalairport,
            f.firstseen,
            f.lastseen,
            (f.lastseen - f.firstseen) as flight_duration,
            f.track
        FROM
            flights_data4 f
        WHERE
            f.firstseen BETWEEN {start_time} AND {end_time}
        ORDER BY
            f.firstseen
        """.format(
            start_time=start_time,
            end_time=end_time,
        )
