from pathlib import Path
from dataclasses import dataclass, field
import torch
# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "julien_data"

@dataclass
class PathConfig:
    raw_data: Path = DATA_DIR / "raw/accident_flight_states"
    processed_data: Path = DATA_DIR / "processed"
    scalers_dir: Path = BASE_DIR / "state_prediction/model/scalers"
    train_data_dir: Path = BASE_DIR / "state_prediction/model/train_data"
    model_dir: Path = BASE_DIR / "state_prediction/model"

    def get_model_config_path(self, model_type: str) -> Path:
        """Get path for model-specific configuration file"""
        return self.model_dir / f"{model_type}_config.json"
    
    def get_model_weights_path(self, model_type: str) -> Path:
        """Get path for model-specific weights file"""
        return self.model_dir / f"{model_type}_best_model.pth"
    
    def get_loss_plot_path(self, model_type: str) -> Path:
        """Get path for model-specific training loss plot"""
        return self.model_dir / f"{model_type}_training_loss_plot.png"

@dataclass
class DataConfig:
    # From prepare_data.py
    resample_interval: str = '2s'
    input_sequence_length: int = 44  # input_len
    prediction_length: int = 1  # pred_len
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
    nhead: int = 4
    num_encoder_layers: int = 5
    num_decoder_layers: int = 1
    dim_feedforward: int = 1024
    dropout: float = 0.2
    target_dim: int = 6

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
    target_dim: int = 6

# FFNN-specific configuration
@dataclass
class FFNNConfig:
    hidden_dims: list = field(
        default_factory=lambda: [512, 256, 128]
    )
    dropout: float = 0.3
    target_dim: int = 6

# Kalman Filter-specific configuration
@dataclass
class KalmanConfig:
    state_dim: int = None       # If None, will default to 2*input_dim
    process_noise: float = 1e-4
    measurement_noise: float = 1e-2
    dt: float = 3.0             # Time step matching your 3s resampling interval
    target_dim: int = 6

# General model parameters: common parameters and nested model-specific configurations
@dataclass
class ModelConfig:
    input_dim: int = 6  # Common to all models
    transformer: TransformerConfig = TransformerConfig()
    lstm: LSTMConfig = LSTMConfig()
    ffnn: FFNNConfig = FFNNConfig()
    kalman: KalmanConfig = KalmanConfig()
    model_type: str = "transformer"  # Default model type

    @property
    def model_filename(self) -> str:
        """Dynamically generate model filename based on model type."""
        return f"{self.model_type.lower()}_best_model.pth"
    
    def get_model_params(self) -> dict:
        """Get the parameters for the current model type"""
        if self.model_type.lower() == "transformer":
            return {
                "input_dim": self.input_dim,
                **vars(self.transformer)
            }
        elif self.model_type.lower() == "lstm":
            return {
                "input_dim": self.input_dim,
                **vars(self.lstm)
            }
        elif self.model_type.lower() == "ffnn":
            return {
                "input_dim": self.input_dim,
                **vars(self.ffnn)
            }
        elif self.model_type.lower() == "kalman":
            return {
                "input_dim": self.input_dim,
                **vars(self.kalman)
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
    def get_model_dir(self, model_type):
        """Get the directory for a specific model type"""
        if model_type.lower() == "transformer":
            return self.TRANSFORMER_MODEL_DIR
        elif model_type.lower() == "lstm":
            return self.LSTM_MODEL_DIR
        elif model_type.lower() == "ffnn":
            return self.FFNN_MODEL_DIR
        else:
            raise ValueError(f"Unknown model type: {model_type}")

@dataclass 
class TrainingConfig:
    batch_size: int = 64         # Increased from 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    save_model: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Early stopping parameters
    patience: int = 15           # Number of epochs to wait for improvement
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
