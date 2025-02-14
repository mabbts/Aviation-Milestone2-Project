from src.utils.constants import GEORGIA_BOUNDS

class StateVectorQueries:
    """
    Queries for state vector data.
    """
    @staticmethod
    def get_state_vectors(start_time: int, end_time: int, bounds: dict = GEORGIA_BOUNDS) -> str:
        """
        Get state vector data for aircraft within a time range and within specified geographic bounds.

        Args:
            start_time: Unix timestamp for start time.
            end_time: Unix timestamp for end time.
            bounds: Dictionary specifying the geographic bounds with keys "north", "south", "west", "east".
                    If None, defaults to GEORGIA_BOUNDS.

        Returns:
            SQL query string that retrieves state vector data including position, velocity, altitude,
            and other flight parameters constrained by both time and geographic location.
        """

        return f"""
        SELECT
            icao24,
            time,
            lat,
            lon, 
            velocity,
            heading,
            vertrate,
            callsign,
            onground,
            spi,
            squawk,
            geoaltitude,
            baroaltitude
        FROM
            state_vectors_data4
        WHERE
            time BETWEEN {start_time} AND {end_time}
            AND lat BETWEEN {bounds['south']} AND {bounds['north']}
            AND lon BETWEEN {bounds['west']} AND {bounds['east']}
        ORDER BY
            time
        """

    @staticmethod
    def get_state_vectors_by_icao(icao: str, start_time: int = None, end_time: int = None) -> str:
        """
        Retrieve state vector data for a specific aircraft identified by its ICAO24 code.
        Optionally, filter the results within a specified time interval if start_time and end_time are provided.

        Args:
            icao: The unique ICAO24 identifier for the aircraft.
            start_time: (optional) Unix timestamp for the beginning of the time interval.
            end_time: (optional) Unix timestamp for the end of the time interval.

        Returns:
            SQL query string that retrieves state vector data for the specified aircraft,
            including an optional time filter if timestamps are provided.
        """
        time_clause = ""
        if start_time is not None and end_time is not None:
            time_clause = f" AND time BETWEEN {start_time} AND {end_time}"

        return f"""
        SELECT
            icao24,
            time,
            lat,
            lon, 
            velocity,
            heading,
            vertrate,
            callsign,
            onground,
            spi,
            squawk,
            geoaltitude,
            baroaltitude
        FROM
            state_vectors_data4
        WHERE
            icao24 = '{icao}'
            {time_clause}
        ORDER BY
            time
        """