"""
PyOpenSky and Trino Database Usage Guide
---------------------------------------

This guide demonstrates common usage patterns for accessing flight data
using the PyOpenSky library and Trino database.
"""

from pyopensky.trino import Trino
from datetime import datetime, timedelta

def basic_usage_examples():
    """Basic examples of PyOpenSky Trino usage."""
    
    # Initialize Trino connection
    trino = Trino()
    
    # Example 1: Get flight list for a specific airport
    def get_airport_flights(airport_code: str, start_time: datetime):
        """Get flights for a specific airport in the last 24 hours."""
        flights = trino.flightlist(
            start=start_time,
            stop=start_time + timedelta(days=1),
            airport=airport_code
        )
        return flights
  
    # Example 2: Track specific aircraft
    def track_aircraft(icao24: str, start_time: datetime):
        """Get flight history for a specific aircraft."""
        history = trino.history(
            start=start_time,
            stop=start_time + timedelta(hours=6),
            icao24=icao24
        )
        return history

    # Example 3: Get raw data for geographical area
    def get_area_data(bounds: tuple, start_time: datetime):
        """Get raw flight data for a geographical area.
        bounds format: (west, south, east, north)
        """
        raw_data = trino.rawdata(
            start=start_time,
            stop=start_time + timedelta(hours=1),
            bounds=bounds
        )
        return raw_data

def advanced_usage_examples():
    """Advanced query examples with multiple parameters."""
    
    trino = Trino()
    
    # Example 4: Complex flight search
    def search_specific_flights(start_time: datetime):
        """Search flights with multiple criteria."""
        flights = trino.flightlist(
            start=start_time,
            stop=start_time + timedelta(hours=12),
            departure_airport="EDDF",  # Frankfurt
            arrival_airport="EGLL",    # London Heathrow
            cached=False,              # Force fresh data
            compress=True              # Compress cached results
        )
        return flights

    # Example 5: Historical analysis
    def analyze_airport_traffic(airport_code: str, start_time: datetime):
        """Analyze airport traffic with extended parameters."""
        traffic = trino.history(
            start=start_time,
            stop=start_time + timedelta(days=1),
            airport=airport_code,
            time_buffer="2h",          # Include flights +/- 2 hours
            selected_columns=(         # Specify needed columns
                "icao24",
                "callsign",
                "latitude",
                "longitude",
                "altitude"
            )
        )
        return traffic

def usage_tips():
    """
    Important tips for using PyOpenSky:
    
    1. Cache Management:
       - Use cached=True (default) for repeated queries
       - Set cached=False for real-time data
       - Use compress=True for large datasets
    
    2. Time Management:
       - Always use UTC times
       - Split long time ranges into smaller chunks
       - Use time_buffer for airport queries
    
    3. Query Optimization:
       - Specify selected_columns when possible
       - Use appropriate filters (airport, bounds, icao24)
       - Consider using limit parameter for testing
    """
    pass

if __name__ == "__main__":
    # Example usage
    now = datetime.utcnow()
    
    trino = Trino()
    
    # Get flights from Frankfurt in the last 24 hours
    frankfurt_flights = trino.flightlist(
        start=now - timedelta(days=1),
        airport="EDDF"
    )
    
    print(f"Found {len(frankfurt_flights) if frankfurt_flights is not None else 0} flights") 