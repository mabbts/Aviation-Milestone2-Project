from typing import List, Optional, Union

class FlightQueries:
    """
    A class containing static methods for generating SQL queries related to flight data.
    """
    @staticmethod
    def get_flight_data_v4(start_time: int, end_time: int, airports: Optional[List[str]] = None) -> str:
        """
        Generates an SQL query to retrieve flight data from the flights_data4 table.

        This query selects flight records within a specified time range, optionally filtering by a list of airports.
        It retrieves the icao24, callsign, estimated departure airport, estimated arrival airport,
        first seen time, last seen time, and track information.

        Args:
            start_time (int): The start time (Unix timestamp) for the data retrieval.
            end_time (int): The end time (Unix timestamp) for the data retrieval.
            airports (Optional[List[str]]): An optional list of airport ICAO codes to filter the results.
                                            If provided, the query will only return flights that either departed from
                                            or arrived at one of the specified airports.

        Returns:
            str: An SQL query string.
        """
        airport_condition = ""
        if airports:
            # Prepare a comma-separated string of quoted airport ICAO codes.
            airport_list = ', '.join(f"'{airport}'" for airport in airports)
            airport_condition = f" AND (estdepartureairport IN ({airport_list}) OR estarrivalairport IN ({airport_list}))"
        
        return f"""
        SELECT
            icao24,
            callsign,
            estdepartureairport,
            estarrivalairport,
            firstseen,
            lastseen,
            track
        FROM flights_data4
        WHERE firstseen BETWEEN {start_time} AND {end_time}{airport_condition}
        ORDER BY firstseen
        """

    @staticmethod
    def get_flight_data_v5(start_time: int, end_time: int, airports: Optional[List[str]] = None) -> str:
        """
        Generates an SQL query to retrieve flight data from the flights_data5 table.

        This query selects flight records within a specified time range, optionally filtering by a list of airports.
        It retrieves the icao24, callsign, departure airport, destination airport, takeoff time,
        landing time, and track information.

        Args:
            start_time (int): The start time (Unix timestamp) for the data retrieval.
            end_time (int): The end time (Unix timestamp) for the data retrieval.
            airports (Optional[List[str]]): An optional list of airport ICAO codes to filter the results.
                                            If provided, the query will only return flights that either departed from
                                            or arrived at one of the specified airports.

        Returns:
            str: An SQL query string.
        """
        airport_condition = ""
        if airports:
            # Prepare a comma-separated string of quoted airport ICAO codes.
            airport_list = ', '.join(f"'{airport}'" for airport in airports)
            airport_condition = f" AND (airportofdeparture IN ({airport_list}) OR airportofdestination IN ({airport_list}))"
        
        return f"""
        SELECT
            icao24,
            callsign,
            airportofdeparture,
            airportofdestination,
            takeofftime,
            landingtime,
            track
        FROM flights_data5
        WHERE takeofftime BETWEEN {start_time} AND {end_time}{airport_condition}
        ORDER BY takeofftime
        """

    @staticmethod
    def get_flight_data_by_icao(icao: Union[str, List[str]]) -> str:
        """
        Generates an SQL query to retrieve flight data from the flights_data4 table filtering
        by a single ICAO number or a list of ICAO numbers.

        This query selects all flight records from flights_data4 where the icao24 field exactly matches
        the provided ICAO number, or if a list is provided, where the icao24 field is in the given list.
        The results are ordered by the 'firstseen' timestamp.

        Args:
            icao (Union[str, List[str]]): A single ICAO24 identifier or a list of ICAO24 identifiers.

        Returns:
            str: An SQL query string.

        Raises:
            ValueError: If an empty list is provided.
        """
        if isinstance(icao, list):
            if not icao:
                raise ValueError("The list of ICAO identifiers cannot be empty.")
            # Build a comma-separated list of quoted ICAO values.
            icao_list_str = ", ".join([f"'{item}'" for item in icao])
            condition = f"icao24 IN ({icao_list_str})"
        else:
            condition = f"icao24 = '{icao}'"

        return f"""
        SELECT
            icao24,
            callsign,
            estdepartureairport,
            estarrivalairport,
            firstseen,
            lastseen,
            track
        FROM flights_data4
        WHERE {condition}
        ORDER BY firstseen
        """
