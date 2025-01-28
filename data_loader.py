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
                 compression_enabled: bool = False):
        """
        Initialize loader with configuration
        
        Args:
            config_path: Path to YAML config file with region definitions
            default_region: Key for default region to use
            time_format: Default datetime format for querying
            cache_enabled: Whether to enable query caching
            compression_enabled: Whether to enable data compression
        """
        self.trino = Trino()
        self.time_format = time_format
        self.cache_enabled = cache_enabled
        self.compression_enabled = compression_enabled
        self.regions = self._load_config(config_path)
        self.default_region = default_region
        self.default_bounds = self._get_region_bounds(default_region) if default_region else None

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

    def sample_flights_with_vectors(
        self,
        start_time: Union[datetime, str],
        end_time: Optional[Union[datetime, str]] = None,
        n_samples: int = 10,
        airport: Optional[str] = None,
        include_vectors: bool = True,
        vector_time_buffer: Optional[int] = 300  # 5 minutes buffer in seconds
    ) -> Dict[str, pd.DataFrame]:
        """
        Sample flights and optionally get their corresponding state vectors
        
        Args:
            start_time: Start time for query
            end_time: End time for query (optional)
            n_samples: Number of flights to sample
            airport: Filter by either departure or arrival airport
            include_vectors: Whether to include state vectors
            vector_time_buffer: Time buffer in seconds to look for state vectors before/after flight times
            
        Returns:
            Dictionary containing 'flights' DataFrame and optionally 'state_vectors' DataFrame
        """
        # Get flight samples
        flights = self.load_flights(
            start_time=start_time,
            end_time=end_time,
            airport=airport
        )
        
        if flights.empty:
            logger.warning("No flights found for the given criteria")
            return {'flights': pd.DataFrame()}
            
        # Sample flights
        sampled_flights = flights.sample(n=min(n_samples, len(flights)))
        
        result = {'flights': sampled_flights}
        
        # Get corresponding state vectors if requested
        if include_vectors:
            all_vectors = []
            
            for _, flight in sampled_flights.iterrows():
                # Get flight times
                firstseen = flight.get('firstseen')
                lastseen = flight.get('lastseen')
                
                if firstseen and lastseen:
                    # Add buffer to search window
                    vector_start = firstseen - pd.Timedelta(seconds=vector_time_buffer)
                    vector_end = lastseen + pd.Timedelta(seconds=vector_time_buffer)
                    
                    # Get state vectors for this flight
                    vectors = self.load_state_vectors(
                        start_time=vector_start,
                        end_time=vector_end,
                        icao24=flight.get('icao24'),
                        callsign=flight.get('callsign')
                    )
                    
                    if not vectors.empty:
                        # Add flight identifier columns
                        vectors['flight_id'] = flight.get('flight_id', '')
                        vectors['firstseen'] = firstseen
                        vectors['lastseen'] = lastseen
                        all_vectors.append(vectors)
            
            if all_vectors:
                result['state_vectors'] = pd.concat(all_vectors, ignore_index=True)
            else:
                result['state_vectors'] = pd.DataFrame()
                logger.warning("No state vectors found for the sampled flights")
        
        return result

# Usage Example
if __name__ == "__main__":
    loader = OpenSkyLoader(config_path='config.yaml', default_region='georgia')
    try:
        # Sample flights with their state vectors
        print("\nSampling flights with state vectors...")
        sampled_data = loader.sample_flights_with_vectors(
            start_time='2024-03-01 08:00:00',
            end_time='2024-03-01 09:00:00',
            n_samples=5,
            airport='KATL',  # Atlanta International
            vector_time_buffer=300  # 5 minutes buffer
        )
        
        if not sampled_data['flights'].empty:
            print(f"\nSampled {len(sampled_data['flights'])} flights")
            print("Sample flights data:\n", sampled_data['flights'][['icao24', 'callsign', 'firstseen', 'lastseen']].head())
            
            if 'state_vectors' in sampled_data and not sampled_data['state_vectors'].empty:
                print(f"\nFound {len(sampled_data['state_vectors'])} state vectors")
                print("Sample state vectors:\n", sampled_data['state_vectors'][['time', 'icao24', 'callsign', 'lat', 'lon']].head())
            
    except Exception as e:
        print(f"Data loading failed: {str(e)}")