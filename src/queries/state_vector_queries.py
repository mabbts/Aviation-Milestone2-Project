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