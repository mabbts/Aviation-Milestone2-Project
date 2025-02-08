"""
This script samples state vector data and saves the samples to a specified directory.
It uses the StateVectorPipeline to fetch the data.
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.pipeline.state_vector_pipeline import StateVectorPipeline
from src.utils.paths import DATA_DIR


StateVectorPipeline.sample_state_vectors(
    start_date="2024-01-01 12:00:00",
    end_date="2024-12-31 15:00:00",
    output_dir=DATA_DIR / "raw/state_vectors_samples",
    n_samples=15,
    interval_hours=3,
    skip_if_exists=True
)