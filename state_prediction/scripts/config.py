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

# Transformer-specific configuration.
@dataclass
class TransformerConfig:
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 5
    num_decoder_layers: int = 1
    dim_feedforward: int = 512
    dropout: float = 0.2
    target_dim: int = 7

# LSTM-specific configuration.
@dataclass
class LSTMConfig:
    # width and height of the LSTM
    hidden_dim: int = 256
    num_layers: int = 3
    # regularization parameter
    dropout: float = 0.3
    l2_weight_decay: float = 1e-4
    # additional parameters
    bidirectional: bool = False
    # output dimensions
    target_dim: int = 7

# FFNN-specific configuration
@dataclass
class FFNNConfig:
    hidden_dims: list = field(
        default_factory=lambda: [512, 256, 128]
    )
    dropout: float = 0.3
    target_dim: int = 7

# General model parameters: common parameters and nested model-specific configurations
@dataclass
class ModelConfig:
    model_type: str = "transformer"  # Options: "transformer", "lstm", or "ffnn"
    input_dim: int = 7  # Common to all models
    transformer: TransformerConfig = TransformerConfig()
    lstm: LSTMConfig = LSTMConfig()
    ffnn: FFNNConfig = FFNNConfig()

    @property
    def model_filename(self) -> str:
        """Dynamically generate model filename based on model type."""
        return f"{self.model_type.lower()}_best_model.pth"

@dataclass 
class TrainingConfig:
    batch_size: int = 64         # Increased from 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    save_model: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Early stopping parameters
    patience: int = 10           # Number of epochs to wait for improvement
    min_delta: float = 1e-4      # Minimum change to qualify as an improvement
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6

@dataclass
class InferenceConfig:
    num_generated_steps: int = 10  # For sequence generation

# Instantiate configurations
PATHS = PathConfig()
DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainingConfig()
INFERENCE = InferenceConfig()
