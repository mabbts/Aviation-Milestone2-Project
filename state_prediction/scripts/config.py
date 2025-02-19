from pathlib import Path
from dataclasses import dataclass, field
import torch
# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"

@dataclass
class PathConfig:
    raw_data: Path = DATA_DIR / "raw/accident_flight_states"
    processed_data: Path = DATA_DIR / "processed"
    scalers_dir: Path = BASE_DIR / "state_prediction/model/scalers"
    train_data_dir: Path = BASE_DIR / "state_prediction/model/train_data"
    model_dir: Path = BASE_DIR / "state_prediction/model"

@dataclass
class DataConfig:
    # From prepare_data.py
    resample_interval: str = '3s'
    input_sequence_length: int = 29  # input_len
    prediction_length: int = 1       # pred_len
    feature_columns: list = field(
        default_factory=lambda: [
            'lon', 'lat', 'heading', 'velocity', 
            'vertrate', 'geoaltitude'
        ]
    )
    target_columns: list = field(
        default_factory=lambda: [
            'lon', 'lat', 'heading', 'velocity', 
            'vertrate', 'geoaltitude'
        ]
    )
    test_size: float = 0.2
    random_state: int = 42
    interpolation_columns: list = field(
        default_factory=lambda: [
            'lat', 'lon', 'velocity', 'heading',
            'vertrate', 'geoaltitude', 'baroaltitude'
        ]
    )

@dataclass
class ModelConfig:
    model_type: str = "transformer"  # Added field for factory selection
    input_dim: int = 7
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 1
    dim_feedforward: int = 1024
    dropout: float = 0.3
    target_dim: int = 7

@dataclass 
class TrainingConfig:
    # From train.py
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    save_model: bool = True
    model_filename: str = 'best_model.pth'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class InferenceConfig:
    num_generated_steps: int = 10  # For sequence generation

# Instantiate configurations
PATHS = PathConfig()
DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainingConfig()
INFERENCE = InferenceConfig()
