from pathlib import Path

# Get the project root directory (2 levels up from this file)
ROOT_DIR = Path(__file__).parent.parent.parent

# Define data directory path
DATA_DIR = ROOT_DIR / "data"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)