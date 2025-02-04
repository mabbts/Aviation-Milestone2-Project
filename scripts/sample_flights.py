import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.flight_pipeline import FlightsPipeline
from src.utils.paths import DATA_DIR

FlightsPipeline.sample_flight_v4(
    start_date="2025-01-01",
    end_date="2025-12-31",
    output_dir=DATA_DIR / "raw/v4_samples",
    n_samples=10,
    interval_hours=1,
    skip_if_exists=True
)