from typing import List, Optional

class FlightQueries:
    @staticmethod
    def get_flight_data_v4(start_time: int, end_time: int, airports: Optional[List[str]] = None) -> str:
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
