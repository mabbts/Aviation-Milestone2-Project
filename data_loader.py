"""
OpenSky Trino Data Loader and Processor
---------------------------------------
Centralized class for querying and processing aviation data from OpenSky's Trino database.
Handles multiple table types, time-based chunking, and common preprocessing tasks.
"""

from pyopensky.trino import Trino
from datetime import datetime, timedelta, timezone
import pandas as pd
import logging
from typing import Optional, Dict, Union, List, Any
from haversine import haversine, Unit
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSkyLoader:
    """Main class for handling OpenSky data operations"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 default_region: Optional[str] = None,
                 time_format: str = "%Y-%m-%d %H:%M:%S",
                 chunk_size: int = 1,
                 max_retries: int = 3):
        """
        Initialize loader with configuration
        
        Args:
            config_path: Path to YAML config file
            default_region: Key for default region to use
            time_format: Default datetime format for querying
            chunk_size: Default hours per chunk for time-based queries
            max_retries: Maximum number of retries for chunked loading
        """
        self.trino = Trino()
        self.time_format = time_format
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.regions = self._load_config(config_path)
        self.default_region = default_region or self.regions.get('default_region')
        self.default_bounds = self._get_region_bounds(self.default_region)
        self._validate_bounds()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not config_path:
            config_path = Path(__file__).parent / 'config.yaml'
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate regions structure
            for region, bounds in config.get('regions', {}).items():
                self._validate_region(region, bounds)
                
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {str(e)}")

    def _validate_region(self, region_name: str, bounds: Dict):
        """Validate a single region configuration"""
        required = ['north', 'south', 'west', 'east']
        if not all(k in bounds for k in required):
            raise ValueError(f"Invalid configuration for region '{region_name}'. "
                             f"Must contain {required}")
        self._validate_bounds(bounds)

    def _get_region_bounds(self, region_name: str) -> Dict:
        """Get bounds for a named region"""
        if region_name not in self.regions.get('regions', {}):
            available = list(self.regions.get('regions', {}).keys())
            raise ValueError(f"Region '{region_name}' not found. Available regions: {available}")
        return self.regions['regions'][region_name]

    def _validate_bounds(self):
        """Ensure bounds are valid geographic coordinates"""
        required = ['north', 'south', 'west', 'east']
        if not all(k in self.default_bounds for k in required):
            raise ValueError("Bounds must contain north, south, west, east")
            
    def _get_table_config(self, table_name: str) -> Dict:
        """Return column mappings and query templates for different tables"""
        configs = {
            'state_vectors_data4': {
                'columns': [
                    'time', 'icao24', 'callsign', 'lat', 'lon',
                    'geoaltitude', 'velocity', 'heading', 'vertrate', 'onground'
                ],
                'rename_map': {
                    'lat': 'latitude',
                    'lon': 'longitude',
                    'geoaltitude': 'altitude'
                },
                'partition_key': 'hour'
            },
            'flights_data4': {
                'columns': [
                    'icao24', 'firstseen', 'estdepartureairport', 'lastseen',
                    'estarrivalairport', 'callsign', 'track'
                ],
                'rename_map': {},
                'partition_key': 'day'
            },
            'position_data4': {
                'columns': [
                    'icao24', 'time', 'lat', 'lon', 'altitude', 
                    'groundspeed', 'track', 'verticalrate'
                ],
                'rename_map': {
                    'lat': 'latitude',
                    'lon': 'longitude'
                },
                'partition_key': 'hour'
            },
            'identification_data4': {
                'columns': [
                    'icao24', 'time', 'typecode', 'registration',
                    'manufacturername', 'model', 'operator'
                ],
                'rename_map': {},
                'partition_key': 'hour'
            }
        }
        return configs.get(table_name, {})
    
    def load_data(
        self,
        table_name: str,
        start_time: Union[datetime, str],
        end_time: Optional[Union[datetime, str]] = None,
        duration_hours: Optional[int] = None,
        bounds: Optional[Dict[str, float]] = None,
        additional_columns: Optional[List[str]] = None,
        join_tables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Main method to load data from OpenSky database
        """
        # Validate table exists
        self._validate_table_access(table_name)
        
        # Time handling and validation
        start_time, end_time = self._parse_time_params(start_time, end_time, duration_hours)
        
        # Get table-specific configuration
        config = self._get_table_config(table_name)
        
        # Build query parameters
        columns = config.get('columns', []) + (additional_columns or [])
        
        # Validate columns exist
        self._validate_columns(table_name, columns)
        
        # Use provided bounds or default
        query_bounds = bounds or self.default_bounds
        
        # Execute in chunks if needed
        if (end_time - start_time).total_seconds() / 3600 > self.chunk_size:
            return self._chunked_load(table_name, start_time, end_time, columns, query_bounds)
            
        return self._execute_query(table_name, columns, start_time, end_time, query_bounds)

    def _parse_time_params(self, start_time, end_time, duration_hours):
        """Validate and parse time parameters with timezone awareness"""
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, self.time_format).replace(tzinfo=timezone.utc)
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, self.time_format).replace(tzinfo=timezone.utc)
        if duration_hours:
            end_time = start_time + timedelta(hours=duration_hours)
            
        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)
            
        if end_time <= start_time:
            raise ValueError("End time must be after start time")
            
        return start_time, end_time

    def _chunked_load(self, table_name, start_time, end_time, columns, bounds):
        """Handle large time ranges by splitting into manageable chunks"""
        chunk_duration = timedelta(hours=self.chunk_size)
        chunks = pd.date_range(start_time, end_time, freq=chunk_duration, inclusive='both')
        
        results = []
        for chunk_start in chunks:
            chunk_end = min(chunk_start + chunk_duration, end_time)
            for attempt in range(self.max_retries):
                try:
                    df = self._execute_query(table_name, columns, chunk_start, chunk_end, bounds)
                    results.append(df)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed chunk {chunk_start} after {self.max_retries} attempts")
                        raise
                    logger.warning(f"Retrying chunk {chunk_start} (attempt {attempt+1})")
                    
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def _execute_query(self, table_name: str, columns: List[str], 
                      start_time: datetime, end_time: datetime, bounds: Dict):
        """Execute a single Trino query with partition optimization"""
        try:
            config = self._get_table_config(table_name)
            partition_key = config.get('partition_key')
            
            # Build partition filters
            partition_filters = []
            if partition_key == 'hour':
                start_hour = start_time.hour
                end_hour = end_time.hour
                partition_filters.append(f"hour >= {start_hour} AND hour <= {end_hour}")
            elif partition_key == 'day':
                start_day = start_time.day
                end_day = end_time.day
                partition_filters.append(f"day >= {start_day} AND day <= {end_day}")
            
            df = self.trino.history(
                start=start_time,
                stop=end_time,
                bounds=(
                    bounds['west'], 
                    bounds['south'], 
                    bounds['east'], 
                    bounds['north']
                ),
                selected_columns=columns,
                extra_filters=partition_filters,
                cached=True,
                compress=True
            )
            return self._postprocess(df, table_name, bounds)
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def _postprocess(self, df: pd.DataFrame, table_name: str, bounds: Dict) -> pd.DataFrame:
        """Common postprocessing steps with correct bounds handling"""
        if df.empty:
            return df
            
        # Apply table-specific renaming
        config = self._get_table_config(table_name)
        df = df.rename(columns=config.get('rename_map', {}))
        
        # Geographic filtering with correct bounds
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = self._filter_geographic(df, bounds)
            
        # Add aircraft info if available
        if 'icao24' in df.columns:
            df = self._add_aircraft_info(df)
            
        # Handle complex columns
        if 'track' in df.columns:
            df = self._process_track_data(df)
            
        return df

    def _filter_geographic(self, df: pd.DataFrame, bounds: Dict) -> pd.DataFrame:
        """Filter DataFrame to specified bounds"""
        return df.query(
            "latitude >= @bounds['south'] and "
            "latitude <= @bounds['north'] and "
            "longitude >= @bounds['west'] and "
            "longitude <= @bounds['east']"
        )

    def _add_aircraft_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join aircraft metadata from identification_data4"""
        try:
            aircraft_info = self.trino.query("""
                SELECT DISTINCT icao24, typecode, manufacturername as manufacturer, 
                              model, operator, registration
                FROM identification_data4
                WHERE icao24 IN (SELECT DISTINCT icao24 FROM unnest(%(icao_list)s) as t(icao24))
            """, {'icao_list': df['icao24'].unique().tolist()})
            
            return df.merge(aircraft_info, on='icao24', how='left')
        except Exception as e:
            logger.warning(f"Failed to add aircraft info: {str(e)}")
            return df

    def _process_track_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process complex track column into usable format"""
        if 'track' not in df.columns:
            return df
        
        try:
            # Assuming track is a JSON/array string
            df['track'] = df['track'].apply(lambda x: pd.json_normalize(x) if x else None)
            
            # Optionally expand track data into separate columns
            track_columns = ['track_latitude', 'track_longitude', 'track_altitude']
            for col in track_columns:
                df[col] = df['track'].apply(lambda x: x.get(col.replace('track_', '')) if x is not None else None)
            
            return df
        except Exception as e:
            logger.warning(f"Failed to process track data: {str(e)}")
            return df

    def _validate_table_access(self, table_name: str):
        """Check if table exists in Trino catalog"""
        available_tables = self.trino.show_tables()
        if table_name not in available_tables:
            raise ValueError(f"Table {table_name} not found. Available tables: {available_tables}")

    def _validate_columns(self, table_name: str, columns: List[str]):
        """Ensure requested columns exist in target table"""
        table_schema = self.trino.get_columns(table_name)
        existing_columns = [col.name for col in table_schema]
        missing = set(columns) - set(existing_columns)
        if missing:
            raise ValueError(f"Columns {missing} not found in {table_name}")

    def _join_datasets(self, main_df: pd.DataFrame, join_tables: List[str]) -> pd.DataFrame:
        """Join additional tables to main dataset"""
        for join_table in join_tables:
            join_config = self._get_table_config(join_table)
            join_cols = ['icao24', 'time'] if 'time' in join_config['columns'] else ['icao24']
            join_data = self.load_data(join_table, main_df['time'].min(), main_df['time'].max())
            
            main_df = main_df.merge(
                join_data,
                on=join_cols,
                how='left',
                suffixes=('', f'_{join_table}')
            )
        return main_df

# Usage Example
if __name__ == "__main__":
    loader = OpenSkyLoader(chunk_size=4, max_retries=2)
    
    try:
        # Get flight data with aircraft info and position data
        df = loader.load_data(
            config_path='config.yaml',
            default_region='georgia',
            table_name='state_vectors_data4',
            start_time='2024-03-01 08:00:00',
            duration_hours=24,
            additional_columns=['squawk', 'alert'],
            join_tables=['position_data4']
        )
        
        if not df.empty:
            print(f"Loaded {len(df)} records")
            print("Columns available:", df.columns.tolist())
            print("Sample data:\n", df.head(3))
        else:
            print("No data found for the specified parameters")
            
    except Exception as e:
        print(f"Data loading failed: {str(e)}") 