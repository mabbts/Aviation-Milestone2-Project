"""
Test script for OpenSky Trino database queries
Demonstrates usage of flightlist and history methods
"""

from pyopensky.trino import Trino
from datetime import datetime, timedelta
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_flight_queries():
    # Initialize Trino connection
    trino = Trino()
    
    # Define time window (1-hour period)
    start_time = datetime(2024, 1, 1, 9, 0, 0)  # 2024-01-01 08:00:00 UTC
    end_time = start_time + timedelta(hours=1)
    
    try:
        # 1. First get flight list for Atlanta airport
        logger.info("Fetching flight list...")
        flights = trino.flightlist(
            start=start_time,
            stop=end_time,
            airport='KATL',  # Atlanta International
            cached=True,
            compress=True,
            limit=10  # Limit to 10 flights for testing
        )
        
        if flights is not None and not flights.empty:
            logger.info(f"Found {len(flights)} flights")
            logger.info("\nSample flight data:")
            print(flights[['icao24', 'callsign', 'firstseen', 'lastseen', 
                         'departure_airport', 'arrival_airport']].head())
            
            # 2. Get detailed state vectors for the first flight
            if len(flights) > 0:
                sample_flight = flights.iloc[0]
                logger.info(f"\nFetching state vectors for flight: {sample_flight['callsign']}")
                
                # Add small buffer around flight times
                buffer = timedelta(minutes=5)
                flight_start = sample_flight['firstseen'] - buffer
                flight_end = sample_flight['lastseen'] + buffer
                
                states = trino.history(
                    start=flight_start,
                    stop=flight_end,
                    icao24=sample_flight['icao24'],
                    callsign=sample_flight['callsign'],
                    selected_columns=(
                        "time",
                        "icao24",
                        "callsign",
                        "lat",
                        "lon",
                        "geoaltitude",
                        "velocity",
                        "heading",
                        "vertrate",
                        "onground"
                    ),
                    cached=True,
                    compress=True
                )
                
                if states is not None and not states.empty:
                    logger.info(f"Found {len(states)} state vector records")
                    logger.info("\nSample state vector data:")
                    print(states[['time', 'lat', 'lon', 'velocity', 
                                'heading', 'vertrate', 'baroaltitude']].head())
                else:
                    logger.warning("No state vectors found for the flight")
        else:
            logger.warning("No flights found for the specified criteria")
            
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_flight_queries() 