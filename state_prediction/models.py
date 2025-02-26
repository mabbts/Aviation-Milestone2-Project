import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------------------------------------------------
# Positional Encoding for Transformer
# --------------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Positional encoding layer that adds positional information to input embeddings.
    This helps the transformer model understand the order of the sequence.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model (int): The embedding dimension
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length to pre-compute encodings for
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create a long enough P.E. matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Input with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# --------------------------------------------------------------------------------------
# Base Predictor Class
# --------------------------------------------------------------------------------------
class BasePredictor(nn.Module):
    """
    Base class that all predictors should inherit from.
    Provides a common interface for all models.
    """
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

# --------------------------------------------------------------------------------------
# Transformer Predictor
# --------------------------------------------------------------------------------------
class TransformerPredictor(BasePredictor):
    """
    Transformer-based sequence predictor that uses encoder-decoder architecture.
    Particularly effective for capturing long-range dependencies in sequential data.
    """
    def __init__(
        self, 
        input_dim=7, 
        d_model=256, 
        nhead=8, 
        num_encoder_layers=6, 
        num_decoder_layers=1, 
        dim_feedforward=1024, 
        dropout=0.3, 
        target_dim=7
    ):
        """
        Args:
            input_dim (int): Number of input features
            d_model (int): Dimension of the model's internal representations
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of transformer encoder layers
            num_decoder_layers (int): Number of transformer decoder layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
            target_dim (int): Dimension of output predictions
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output layer + learnable target token
        self.fc_out = nn.Linear(d_model, target_dim)
        self.target_embedding = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        """
        Forward pass of the transformer predictor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor: Predictions of shape (batch_size, target_dim)
        """
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)
        decoder_input = self.target_embedding.expand(batch_size, -1, -1)
        decoded = self.transformer_decoder(decoder_input, memory)
        out = self.fc_out(decoded.squeeze(1))
        return out

# --------------------------------------------------------------------------------------
# LSTM Predictor
# --------------------------------------------------------------------------------------
class LSTMPredictor(BasePredictor):
    """
    LSTM-based sequence predictor.
    Effective for capturing temporal dependencies and maintaining long-term memory.
    """
    def __init__(
        self,
        input_dim=7,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        target_dim=7,
        bidirectional=False,
        l2_weight_decay=1e-4
    ):
        """
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Size of LSTM hidden state
            num_layers (int): Number of stacked LSTM layers
            dropout (float): Dropout rate between LSTM layers
            target_dim (int): Dimension of output predictions
            bidirectional (bool): Whether to use bidirectional LSTM
            l2_weight_decay (float): L2 regularization strength
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.l2_weight_decay = l2_weight_decay

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, target_dim)

    def forward(self, x):
        """
        Forward pass of the LSTM predictor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor: Predictions of shape (batch_size, target_dim)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out

# --------------------------------------------------------------------------------------
# Feed-Forward Neural Network Predictor
# --------------------------------------------------------------------------------------
class FFNNPredictor(BasePredictor):
    """
    Simple Feed-Forward Neural Network predictor.
    Flattens the input sequence and processes it through multiple fully connected layers.
    """
    def __init__(
        self,
        input_dim=7,
        seq_len=29,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        target_dim=7
    ):
        """
        Args:
            input_dim (int): Number of input features
            seq_len (int): Length of input sequence
            hidden_dims (list): Dimensions of hidden layers
            dropout (float): Dropout rate between layers
            target_dim (int): Dimension of output predictions
        """
        super().__init__()
        
        self.flat_input_size = input_dim * seq_len
        layer_dims = [self.flat_input_size] + hidden_dims
        
        # Build multi-layer perceptron
        layers = []
        for i in range(len(layer_dims)-1):
            layers.extend([
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        layers.append(nn.Linear(layer_dims[-1], target_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the FFNN predictor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor: Predictions of shape (batch_size, target_dim)
        """
        batch_size = x.size(0)
        expected_features = x.size(1) * x.size(2)
        if expected_features != self.flat_input_size:
            raise ValueError(f"Input shape mismatch. Expected {self.flat_input_size} features when flattened, but got {expected_features}. "
                           f"Input shape: {x.shape}")
        
        x_flat = x.reshape(batch_size, -1)
        return self.model(x_flat)

# --------------------------------------------------------------------------------------
# Kalman Filter Predictor
# --------------------------------------------------------------------------------------
class KalmanFilterPredictor(BasePredictor):
    """
    Kalman Filter-based sequence predictor.
    Effective for sequential state estimation with noisy measurements.
    """
    def __init__(
        self,
        input_dim=7,            # Dimension of the input features
        state_dim=None,         # Dimension of the state vector (defaults to 2x input_dim for position+velocity)
        process_noise=1e-4,     # Process noise covariance factor
        measurement_noise=1e-2, # Measurement noise covariance factor
        dt=3.0,                 # Time step in seconds (matches your 3s resampling)
        target_dim=7            # Dimension of output predictions
    ):
        """
        Args:
            input_dim (int): Number of input features
            state_dim (int): Dimension of internal state (if None, uses 2*input_dim)
            process_noise (float): Process noise covariance factor
            measurement_noise (float): Measurement noise covariance factor
            dt (float): Time delta between measurements in seconds
            target_dim (int): Dimension of output predictions
        """
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = 2 * input_dim if state_dim is None else state_dim
        self.target_dim = target_dim
        self.dt = dt
        
        # Initialize state transition matrix (F)
        self.register_buffer("F", self._build_state_transition_matrix(dt))
        
        # Initialize measurement matrix (H)
        self.register_buffer("H", self._build_measurement_matrix())
        
        # Initialize process noise covariance (Q)
        self.register_buffer("Q", self._build_process_noise_matrix(process_noise))
        
        # Initialize measurement noise covariance (R)
        self.register_buffer("R", self._build_measurement_noise_matrix(measurement_noise))
        
        # Output projection if needed
        self.output_projection = nn.Linear(self.state_dim, target_dim)
    
    def _build_state_transition_matrix(self, dt):
        """Build the state transition matrix for the Kalman filter."""
        # For a simple constant velocity model
        F = torch.eye(self.state_dim)
        # Connect position to velocity (x += vx * dt)
        for i in range(self.input_dim):
            F[i, i + self.input_dim] = dt
        return F
    
    def _build_measurement_matrix(self):
        """Build the measurement matrix for the Kalman filter."""
        # For direct observation of position only
        H = torch.zeros(self.input_dim, self.state_dim)
        # Observe the first input_dim elements (positions)
        for i in range(self.input_dim):
            H[i, i] = 1.0
        return H
    
    def _build_process_noise_matrix(self, noise_factor):
        """Build the process noise covariance matrix."""
        # Simple diagonal process noise
        return torch.eye(self.state_dim) * noise_factor
    
    def _build_measurement_noise_matrix(self, noise_factor):
        """Build the measurement noise covariance matrix."""
        # Simple diagonal measurement noise
        return torch.eye(self.input_dim) * noise_factor
    
    def _predict_step(self, state, covariance):
        """Kalman filter prediction step with vectorized operations."""
        # For state update: use batch matrix multiplication
        # First reshape state to (batch_size, 1, state_dim)
        batch_size = state.shape[0]
        state_expanded = state.unsqueeze(1)
        
        # Properly expand F to match the batch size
        F_expanded = self.F.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Perform batch matrix multiplication
        new_state = torch.bmm(state_expanded, F_expanded.transpose(1, 2)).squeeze(1)
        
        # For covariance: use batch matrix multiplication
        # F @ covariance @ F.T + Q
        temp = torch.bmm(F_expanded, covariance)
        new_covariance = torch.bmm(temp, F_expanded.transpose(1, 2))
        
        # Add Q to each item in batch
        Q_expanded = self.Q.unsqueeze(0).expand(batch_size, -1, -1)
        new_covariance = new_covariance + Q_expanded
        
        return new_state, new_covariance
        
    def _update_step(self, state, covariance, measurement):
        """Kalman filter update step with vectorized operations for batches."""
        batch_size = state.shape[0]
        
        # Expand matrices for batch operations
        H_expanded = self.H.unsqueeze(0).expand(batch_size, -1, -1)
        R_expanded = self.R.unsqueeze(0).expand(batch_size, -1, -1)
        I_expanded = torch.eye(self.state_dim, device=state.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Calculate innovation: y - H*x
        innovation = measurement - torch.bmm(H_expanded, state.unsqueeze(2)).squeeze(2)
        
        # Calculate innovation covariance: H*P*H' + R
        temp = torch.bmm(H_expanded, covariance)
        innovation_covariance = torch.bmm(temp, H_expanded.transpose(1, 2)) + R_expanded
        
        # Calculate Kalman gain: P*H'*inv(S)
        kalman_gain_temp = torch.bmm(covariance, H_expanded.transpose(1, 2))
        
        # Handle matrix inverse for each sample in batch
        kalman_gain = torch.zeros_like(kalman_gain_temp)
        for i in range(batch_size):
            kalman_gain[i] = torch.mm(
                kalman_gain_temp[i], 
                torch.linalg.inv(innovation_covariance[i])
            )
        
        # Update state: x + K*y
        state_correction = torch.bmm(kalman_gain, innovation.unsqueeze(2)).squeeze(2)
        new_state = state + state_correction
        
        # Update covariance: (I - K*H)*P
        temp = torch.bmm(kalman_gain, H_expanded)
        new_covariance = torch.bmm((I_expanded - temp), covariance)
        
        return new_state, new_covariance
    
    def forward(self, x):
        """
        Forward pass of the Kalman filter predictor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor: Predictions of shape (batch_size, target_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize state with zeros
        state = torch.zeros(batch_size, self.state_dim, device=device)
        # First half is position (from first measurement)
        state[:, :self.input_dim] = x[:, 0, :]
        
        # Initialize covariance with identity for each batch element
        covariance = torch.eye(self.state_dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Process the sequence
        for t in range(seq_len):
            # Prediction step
            state, covariance = self._predict_step(state, covariance)
            
            # Update step (if we have measurements)
            measurement = x[:, t, :]
            state, covariance = self._update_step(state, covariance, measurement)
        
        # Make one final prediction step to get the next state
        next_state, _ = self._predict_step(state, covariance)
        
        # Project to output dimension if necessary
        output = self.output_projection(next_state)
        
        return output

# --------------------------------------------------------------------------------------
# Model Factory
# --------------------------------------------------------------------------------------
def get_model(model_type="transformer", **kwargs):
    """
    Factory function to instantiate the appropriate model type.
    
    Args:
        model_type (str): Type of model to create ("transformer", "lstm", "ffnn", or "kalman")
        **kwargs: Model-specific parameters
    
    Returns:
        BasePredictor: Instantiated model of the requested type
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type.lower() == "transformer":
        return TransformerPredictor(**kwargs)
    elif model_type.lower() == "lstm":
        return LSTMPredictor(**kwargs)
    elif model_type.lower() == "ffnn":
        return FFNNPredictor(**kwargs)
    elif model_type.lower() == "kalman":
        return KalmanFilterPredictor(**kwargs)
    else:
        raise ValueError(f"Model type '{model_type}' not recognized.") 