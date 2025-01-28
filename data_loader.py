"""
OpenSky Trino Data Loader
------------------------
Simplified class for querying aviation data from OpenSky's Trino database.
Focuses on core functionality for state vectors and flight data.
"""

from pyopensky.trino import Trino
from datetime import datetime, timezone
import pandas as pd
import logging
from typing import Optional, Dict, Union, List, Any
import yaml
from pathlib import Path
from sqlalchemy import text
import time  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSkyLoader:
    """Main class for handling OpenSky data operations"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 default_region: Optional[str] = None,
                 time_format: str = "%Y-%m-%d %H:%M:%S",
                 cache_enabled: bool = True,
                 compression_enabled: bool = False,
                 request_delay: float = 1.0,
                 time_buffer: Optional[Union[str, pd.Timedelta]] = None):
        """
        Initialize loader with configuration
        
        Args:
            config_path: Path to YAML config file with region definitions
            default_region: Key for default region to use
            time_format: Default datetime format for querying
            cache_enabled: Whether to enable query caching
            compression_enabled: Whether to enable data compression
            request_delay: Delay in seconds between requests to avoid rate limiting
            time_buffer: Time buffer used to extend time bounds for flights
        """
        self.trino = Trino()
        self.time_format = time_format
        self.cache_enabled = cache_enabled
        self.compression_enabled = compression_enabled
        self.regions = self._load_config(config_path)
        self.default_region = default_region
        self.default_bounds = self._get_region_bounds(default_region) if default_region else None
        self.request_delay = request_delay
        self.time_buffer = time_buffer

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load region definitions from YAML file"""
        if not config_path:
            config_path = Path(__file__).parent / 'config.yaml'
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {str(e)}")
            return {'regions': {}}

    def _get_region_bounds(self, region_name: str) -> Dict:
        """Get bounds for a named region"""
        if not region_name or region_name not in self.regions.get('regions', {}):
            return None
        return self.regions['regions'][region_name]

    def get_flight_data(
        self,
        start_time: Union[datetime, str],
        end_time: Optional[Union[datetime, str]] = None,
        bounds: Optional[Dict[str, float]] = None,
        region: Optional[str] = None,
        departure_airport: Optional[str] = None,
        arrival_airport: Optional[str] = None,
        airport: Optional[str] = None,
        callsign: Optional[str] = None,
        icao24: Optional[str] = None,
        selected_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load flight data and state vectors from OpenSky using the history method
        
        Args:
            start_time: Start time for query
            end_time: End time for query (optional)
            bounds: Manual bounds as dict with north/south/east/west
            region: Named region from config
            departure_airport: Filter by departure airport ICAO
            arrival_airport: Filter by arrival airport ICAO
            airport: Filter by either departure or arrival airport
            callsign: Filter by callsign (wildcards allowed)
            icao24: Filter by aircraft transponder code
            selected_columns: List of columns to retrieve
        """
        time.sleep(self.request_delay)
        
        # Parse times
        start = self._parse_time(start_time)
        stop = self._parse_time(end_time) if end_time else None
        
        # Determine bounds
        query_bounds = None
        if bounds:
            query_bounds = (bounds['west'], bounds['south'], 
                          bounds['east'], bounds['north'])
        elif region:
            region_bounds = self._get_region_bounds(region)
            if region_bounds:
                query_bounds = (region_bounds['west'], region_bounds['south'],
                              region_bounds['east'], region_bounds['north'])
        elif self.default_bounds:
            query_bounds = (self.default_bounds['west'], self.default_bounds['south'],
                          self.default_bounds['east'], self.default_bounds['north'])

        try:
            # If specific columns are requested, convert them to text()
            if selected_columns:
                columns_to_select = [text(col) for col in selected_columns]
            else:
                # When empty, use all columns from StateVectorsData4 table
                columns_to_select = ()
            
            return self.trino.history(
                start=start,
                stop=stop,
                bounds=query_bounds,
                departure_airport=departure_airport,
                arrival_airport=arrival_airport,
                airport=airport,
                callsign=callsign,
                icao24=icao24,
                time_buffer=self.time_buffer,
                cached=self.cache_enabled,
                compress=self.compression_enabled,
                selected_columns=columns_to_select
            )
        except Exception as e:
            logger.error(f"Failed to load flight data: {str(e)}")
            raise

    def _parse_time(self, time_value: Union[datetime, str]) -> datetime:
        """Convert time input to timezone-aware datetime"""
        if isinstance(time_value, str):
            try:
                time_value = datetime.strptime(time_value, self.time_format)
            except ValueError as e:
                raise ValueError(f"Failed to parse time string: {str(e)}")
            
        if time_value.tzinfo is None:
            time_value = time_value.replace(tzinfo=timezone.utc)
        return time_value


# Usage Example
# - snapshot of aircraft activity around Atlanta airport within a specific hour on March 1st, 2024, 
# focusing on flights related to that airport and capturing their state vector data
#  within a slightly extended timeframe of their observed flight duration.
if __name__ == "__main__":
    loader = OpenSkyLoader(
        config_path='config.yaml', 
        default_region='georgia',
        time_buffer='5 minutes'  # Buffer for finding flights
    )
    
    try:
        # Get flights data for Atlanta airport
        print("\nGetting flight data...")
        flight_data = loader.get_flight_data(
            start_time='2024-03-01 08:00:00',
            end_time='2024-03-01 09:00:00',
            airport='KATL',  # Atlanta International
            region='georgia'
        )
        
        if not flight_data.empty:
            print(f"\nFound {len(flight_data)} flight records")
            print("\nSample flight data:")
            print(flight_data[['time', 'icao24', 'callsign', 'lat', 'lon', 'departure_airport', 'arrival_airport']].head())
            
    except Exception as e:
        print(f"Data loading failed: {str(e)}")