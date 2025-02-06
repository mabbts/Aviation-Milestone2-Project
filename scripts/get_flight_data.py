import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.pipeline.flight_pipeline import FlightsPipeline
from src.utils.paths import DATA_DIR

FlightsPipeline.chunked_flight_v4(
    start_date="2025-01-01 12:00:00",
    end_date="2025-01-02 15:00:00",
    output_dir=DATA_DIR / "raw/v4_chunks",
    chunk_hours=1,
    skip_if_exists=True
)
