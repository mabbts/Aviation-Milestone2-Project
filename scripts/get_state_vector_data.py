import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.state_vector_pipeline import StateVectorPipeline
from src.utils.paths import DATA_DIR

StateVectorPipeline.chunked_state_vectors(
    start_date="2025-01-01 12:00:00",
    end_date="2025-01-02 15:00:00",
    output_dir=DATA_DIR / "raw/state_vectors_chunks",
    chunk_hours=1,
    skip_if_exists=True
) 