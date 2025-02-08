"""
This script samples flight data and saves the samples to a specified directory.
It uses the FlightsPipeline to fetch the data.
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.pipeline.flight_pipeline import FlightsPipeline
from src.utils.paths import DATA_DIR

FlightsPipeline.sample_flight_v4(
    start_date="2024-01-01",
    end_date="2024-12-31",
    output_dir=DATA_DIR / "raw/v4_samples",
    n_samples=10,
    interval_hours=1,
    skip_if_exists=True
)