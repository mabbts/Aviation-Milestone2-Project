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
import time  # Add this import at the top

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
                 request_delay: float = 1.0):  # Add request_delay parameter
        """
        Initialize loader with configuration
        
        Args:
            config_path: Path to YAML config file with region definitions
            default_region: Key for default region to use
            time_format: Default datetime format for querying
            cache_enabled: Whether to enable query caching
            compression_enabled: Whether to enable data compression
            request_delay: Delay in seconds between requests to avoid rate limiting
        """
        self.trino = Trino()
        self.time_format = time_format
        self.cache_enabled = cache_enabled
        self.compression_enabled = compression_enabled
        self.regions = self._load_config(config_path)
        self.default_region = default_region
        self.default_bounds = self._get_region_bounds(default_region) if default_region else None
        self.request_delay = request_delay  # Store request_delay

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

    def load_state_vectors(
        self,
        start_time: Union[datetime, str],
        end_time: Optional[Union[datetime, str]] = None,
        bounds: Optional[Dict[str, float]] = None,
        region: Optional[str] = None,
        callsign: Optional[str] = None,
        icao24: Optional[str] = None,
        selected_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load state vectors data from OpenSky
        
        Args:
            start_time: Start time for query
            end_time: End time for query (optional)
            bounds: Manual bounds as dict with north/south/east/west
            region: Named region from config
            callsign: Filter by callsign (wildcards allowed)
            icao24: Filter by aircraft transponder code
            selected_columns: List of columns to retrieve
        """
        time.sleep(self.request_delay)  # Add delay before request
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
            # Use text() for column names to properly reference them in SQL
            default_columns = [
                text(col) for col in [
                    'time', 'icao24', 'callsign', 'lat', 'lon', 
                    'velocity', 'heading', 'geoaltitude'
                ]
            ]
            
            # If specific columns are requested, convert them to text()
            if selected_columns:
                columns_to_select = [text(col) for col in selected_columns]
            else:
                columns_to_select = default_columns
            
            return self.trino.history(
                start=start,
                stop=stop,
                bounds=query_bounds,
                callsign=callsign,
                icao24=icao24,
                cached=self.cache_enabled,
                compress=self.compression_enabled,
                selected_columns=columns_to_select
            )
        except Exception as e:
            logger.error(f"Failed to load state vectors: {str(e)}")
            raise

    def load_flights(
        self,
        start_time: Union[datetime, str],
        end_time: Optional[Union[datetime, str]] = None,
        departure_airport: Optional[str] = None,
        arrival_airport: Optional[str] = None,
        airport: Optional[str] = None,
        callsign: Optional[str] = None,
        icao24: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load flight data from OpenSky
        
        Args:
            start_time: Start time for query
            end_time: End time for query (optional)
            departure_airport: Filter by departure airport ICAO
            arrival_airport: Filter by arrival airport ICAO
            airport: Filter by either departure or arrival airport
            callsign: Filter by callsign (wildcards allowed)
            icao24: Filter by aircraft transponder code
        """
        time.sleep(self.request_delay)  # Add delay before request
        start = self._parse_time(start_time)
        stop = self._parse_time(end_time) if end_time else None

        try:
            return self.trino.flightlist(
                start=start,
                stop=stop,
                departure_airport=departure_airport,
                arrival_airport=arrival_airport,
                airport=airport,
                callsign=callsign,
                icao24=icao24,
                cached=self.cache_enabled,
                compress=self.compression_enabled
            )
        except Exception as e:
            logger.error(f"Failed to load flights: {str(e)}")
            raise

    def _parse_time(self, time_value: Union[datetime, str]) -> datetime:
        """Convert time input to timezone-aware datetime"""
        if isinstance(time_value, str):
            time_value = datetime.strptime(time_value, self.time_format)
        if not time_value.tzinfo:
            time_value = time_value.replace(tzinfo=timezone.utc)
        return time_value

    def get_flights_with_vectors(
        self,
        start_time: Union[datetime, str],
        end_time: Optional[Union[datetime, str]] = None,
        airport: Optional[str] = None,
        bounds: Optional[Dict[str, float]] = None,
        region: Optional[str] = None,
        vector_time_buffer: Optional[int] = 300,  # 5 minutes buffer in seconds
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all flights and their corresponding state vectors efficiently by querying vectors first
        
        Args:
            start_time: Start time for query
            end_time: End time for query (optional)
            airport: Filter by either departure or arrival airport
            bounds: Manual bounds as dict with north/south/east/west
            region: Named region from config to use for bounds
            vector_time_buffer: Time buffer in seconds to look for state vectors before/after flight times
            
        Returns:
            Dictionary containing 'flights' DataFrame and 'state_vectors' DataFrame
        """
        # First get all state vectors for the time period
        logger.info("Fetching state vectors...")
        
        # Determine bounds for the query
        query_bounds = None
        if bounds:
            query_bounds = bounds
        elif region:
            query_bounds = self._get_region_bounds(region)
        elif self.default_bounds:
            query_bounds = self.default_bounds
        
        if query_bounds:
            logger.info(f"Using geographic bounds: N={query_bounds['north']}, S={query_bounds['south']}, "
                       f"E={query_bounds['east']}, W={query_bounds['west']}")
        
        all_vectors = self.load_state_vectors(
            start_time=start_time,
            end_time=end_time,
            bounds=query_bounds,
            region=region if not bounds else None  # Use region only if no manual bounds
        )
        
        if all_vectors.empty:
            logger.warning("No state vectors found for the given time period and bounds")
            return {'flights': pd.DataFrame(), 'state_vectors': pd.DataFrame()}
        
        # Extract unique identifiers from vectors
        unique_icao24s = all_vectors['icao24'].dropna().unique().tolist()
        unique_callsigns = (all_vectors['callsign']
                           .dropna()
                           .str.strip()
                           .replace('', pd.NA)
                           .dropna()
                           .unique()
                           .tolist())
        
        # Get all matching flights in one query
        logger.info(f"Fetching matching flights for {len(unique_icao24s)} unique aircraft...")
        all_flights = self.load_flights(
            start_time=start_time,
            end_time=end_time,
            airport=airport,
            icao24=unique_icao24s if unique_icao24s else None
        )
        
        if all_flights.empty:
            logger.warning("No flights found matching the state vector aircraft")
            return {'flights': pd.DataFrame(), 'state_vectors': pd.DataFrame()}
        
        # Match vectors with flights
        logger.info("Matching vectors with flights...")
        merged_vectors = []
        buffer_td = pd.Timedelta(seconds=vector_time_buffer)
        
        for _, flight in all_flights.iterrows():
            flight_start = flight['firstseen'] - buffer_td
            flight_end = flight['lastseen'] + buffer_td
            
            # Create mask for this flight's vectors
            mask = (
                (all_vectors['time'] >= flight_start) &
                (all_vectors['time'] <= flight_end) &
                (all_vectors['icao24'] == flight['icao24'])
            )
            
            # Add callsign filter if available
            if pd.notna(flight['callsign']):
                mask &= (all_vectors['callsign'] == flight['callsign'].strip())
            
            flight_vectors = all_vectors[mask].copy()
            if not flight_vectors.empty:
                flight_vectors['flight_id'] = flight.get('flight_id', '')
                flight_vectors['firstseen'] = flight['firstseen']
                flight_vectors['lastseen'] = flight['lastseen']
                merged_vectors.append(flight_vectors)
        
        final_vectors = pd.concat(merged_vectors, ignore_index=True) if merged_vectors else pd.DataFrame()
        
        logger.info(f"Found {len(all_flights)} flights and {len(final_vectors)} state vectors")
        return {
            'flights': all_flights,
            'state_vectors': final_vectors
        }

# Usage Example
if __name__ == "__main__":
    loader = OpenSkyLoader(config_path='config.yaml', default_region='georgia')
    try:
        # Sample flights with their state vectors
        print("\nSampling flights with state vectors...")
        sampled_data = loader.get_flights_with_vectors(
            start_time='2024-03-01 08:00:00',
            end_time='2024-03-01 09:00:00',
            airport='KATL',  # Atlanta International
            region='georgia'
        )
        
        if not sampled_data['flights'].empty:
            print(f"\nSampled {len(sampled_data['flights'])} flights")
            print("Sample flights data:\n", sampled_data['flights'][['icao24', 'callsign', 'firstseen', 'lastseen']].head())
            
            if 'state_vectors' in sampled_data and not sampled_data['state_vectors'].empty:
                print(f"\nFound {len(sampled_data['state_vectors'])} state vectors")
                print("Sample state vectors:\n", sampled_data['state_vectors'][['time', 'icao24', 'callsign', 'lat', 'lon']].head())
            
    except Exception as e:
        print(f"Data loading failed: {str(e)}")