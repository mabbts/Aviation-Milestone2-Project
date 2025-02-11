from typing import List, Optional, Union, Tuple

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
    def get_flight_data_by_icao(icao: Union[str, List[str]], time_range: Optional[Union[int, Tuple[int, int]]] = None) -> str:
        """
        Generates an SQL query to retrieve flight data from the flights_data4 table filtering
        by a single ICAO number or a list of ICAO numbers. Optionally, the query can be filtered
        by a specific timestamp or a range of timestamps via the 'firstseen' column.

        This query selects flight records from flights_data4 where the icao24 field exactly matches
        the provided ICAO number, or if a list is provided, where the icao24 field is in the given list.
        If a timestamp is provided as an integer, the results are filtered to only include records matching that time.
        If a tuple is provided, the results are filtered to include records with firstseen between the start 
        and end values (inclusive). The results are ordered by the 'firstseen' timestamp.

        Args:
            icao (Union[str, List[str]]): A single ICAO24 identifier or a list of ICAO24 identifiers.
            time_range (Optional[Union[int, Tuple[int, int]]]): An optional parameter that can be either an integer 
                (to filter for a specific timestamp) or a tuple of two integers (to filter for a range of timestamps).

        Returns:
            str: An SQL query string.

        Raises:
            ValueError: If an empty list is provided or if a tuple is provided for time_range that does not contain exactly two integers.
        """
        if isinstance(icao, list):
            if not icao:
                raise ValueError("The list of ICAO identifiers cannot be empty.")
            # Build a comma-separated list of quoted ICAO values.
            icao_list_str = ", ".join([f"'{item}'" for item in icao])
            icao_condition = f"icao24 IN ({icao_list_str})"
        else:
            icao_condition = f"icao24 = '{icao}'"
        
        time_condition = ""
        if time_range is not None:
            if isinstance(time_range, int):
                time_condition = f" AND firstseen = {time_range}"
            elif isinstance(time_range, tuple):
                if len(time_range) != 2:
                    raise ValueError("When passing a tuple for time_range, it must contain exactly two integers (start_time, end_time).")
                start_time, end_time = time_range
                time_condition = f" AND firstseen BETWEEN {start_time} AND {end_time}"
            else:
                raise ValueError("The time_range parameter must be an int or a tuple of two ints.")

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
        WHERE {icao_condition}{time_condition}
        ORDER BY firstseen
        """
