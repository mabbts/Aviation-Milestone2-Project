import dask
import dask.dataframe as dd
from dask.distributed import Client
import pyarrow as pa
from dask.diagnostics import ProgressBar

def process_flight_data():
    # Initialize local cluster with memory management
    client = Client(n_workers=4, memory_limit='8GB')  # Adjust based on your system
    print(f"Dask Dashboard: {client.dashboard_link}")

    # Load data with explicit schema
    dtypes = {
        'time': 'datetime64[ns]',
        'icao24': 'category',
        'callsign': 'category',
        'latitude': 'float32',
        'longitude': 'float32',
        'altitude': 'float32',
        'velocity': 'float32',
        'heading': 'float32',
        'vertrate': 'float32',
        'onground': 'bool'
    }

    # Read data in chunks
    df = dd.read_parquet(
        "/home/jhovan/GitHub/Aviation-Milestone2-Project/data/georgia_data_2/georgia_complete_dataset.parquet",
        engine='pyarrow',
        dtype=dtypes
    )

    # Process time column in partitions
    df['time'] = df['time'].map_partitions(
        lambda s: dd.to_datetime(s).dt.tz_localize(None),
        meta=('time', 'datetime64[ns]')
    )
    df = df.set_index('time')

    # Resample with optimized aggregations
    resampled = df.resample('1min').agg({
        'icao24': 'first',
        'callsign': 'last',
        'latitude': 'last',
        'longitude': 'last',
        'altitude': 'mean',
        'velocity': 'mean',
        'heading': 'mean',
        'vertrate': 'mean',
        'onground': 'last'
    })

    # Persist the resampled data before saving
    resampled = resampled.persist()

    # Progress bar for monitoring
    with ProgressBar():
        resampled.to_parquet(
            "../../data/georgia_data_2/georgia_resampled_1min.parquet",
            engine='pyarrow',
            schema={
                'time': pa.timestamp('ns'),
                'icao24': pa.string(),
                'callsign': pa.string(),
                'latitude': pa.float32(),
                'longitude': pa.float32(),
                'altitude': pa.float32(),
                'velocity': pa.float32(),
                'heading': pa.float32(),
                'vertrate': pa.float32(),
                'onground': pa.bool_()
            },
            overwrite=True
        )

    client.close()

if __name__ == "__main__":
    with dask.config.set({'temporary_directory': '/tmp/dask-workers'}):
        process_flight_data()