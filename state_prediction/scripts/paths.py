from pathlib import Path

# Get the project root directory (2 levels up from this file)
ROOT_DIR = Path(__file__).parent.parent.parent

# Define data directory path
DATA_DIR = ROOT_DIR / "data"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model paths 
MODEL_DIR = ROOT_DIR / "state_prediction/model"
# transformer model directory path
TRANSFORMER_MODEL_DIR = MODEL_DIR / "transformer"
# lstm model directory path
LSTM_MODEL_DIR = MODEL_DIR / "lstm"
# ffnn model directory path
FFNN_MODEL_DIR = MODEL_DIR / "ffnn"

# Create model directories if they don't exist
TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
FFNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)