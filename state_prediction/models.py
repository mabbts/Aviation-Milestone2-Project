import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# Base class for predictors. In the future, other models can inherit from this.
class BasePredictor(nn.Module):
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method.")


class TransformerPredictor(BasePredictor):
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
        x shape: (batch_size, seq_len, input_dim)
        Returns: (batch_size, target_dim)
        """
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)
        decoder_input = self.target_embedding.expand(batch_size, -1, -1)
        decoded = self.transformer_decoder(decoder_input, memory)
        out = self.fc_out(decoded.squeeze(1))
        return out


class LSTMPredictor(BasePredictor):
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
        LSTM-based predictor.
        Args:
            input_dim (int): Number of features in the input.
            hidden_dim (int): Hidden dimension of the LSTM.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout to apply between LSTM layers.
            target_dim (int): Dimension of the output prediction.
            bidirectional (bool): If True, use a bidirectional LSTM.
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
        x shape: (batch_size, seq_len, input_dim)
        Returns: (batch_size, target_dim)
        """
        # LSTM returns outputs for all time-steps; we take the last time-step.
        lstm_out, (h_n, c_n) = self.lstm(x)
        # If bidirectional, h_n contains the last hidden state for both directions.
        # You can either concatenate them or use the output from the last time-step.
        # Here, we take the output at the final time-step:
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out


class FFNNPredictor(BasePredictor):
    def __init__(
        self,
        input_dim=7,
        seq_len=29,  # Updated to match your actual sequence length (203/7 ≈ 29)
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        target_dim=7
    ):
        """
        Simple Feed-Forward Neural Network predictor.
        Args:
            input_dim (int): Number of features in the input
            seq_len (int): Length of input sequence
            hidden_dims (list): List of hidden layer dimensions
            dropout (float): Dropout rate between layers
            target_dim (int): Dimension of the output prediction
        """
        super().__init__()
        
        # Calculate flattened input size
        self.flat_input_size = input_dim * seq_len
        
        # Create list of layer dimensions
        layer_dims = [self.flat_input_size] + hidden_dims
        
        # Build multi-layer perceptron
        layers = []
        for i in range(len(layer_dims)-1):
            layers.extend([
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Add final output layer
        layers.append(nn.Linear(layer_dims[-1], target_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        Returns: (batch_size, target_dim)
        """
        batch_size = x.size(0)
        # Add shape validation
        expected_features = x.size(1) * x.size(2)
        if expected_features != self.flat_input_size:
            raise ValueError(f"Input shape mismatch. Expected {self.flat_input_size} features when flattened, but got {expected_features}. "
                           f"Input shape: {x.shape}")
        
        # Flatten the input sequence
        x_flat = x.reshape(batch_size, -1)
        return self.model(x_flat)


def get_model(model_type="transformer", **kwargs):
    """
    Factory method to get the desired model.
    To add a new model, implement the new class and update this function.
    """
    if model_type.lower() == "transformer":
        return TransformerPredictor(**kwargs)
    elif model_type.lower() == "lstm":
        return LSTMPredictor(**kwargs)
    elif model_type.lower() == "ffnn":
        return FFNNPredictor(**kwargs)
    else:
        raise ValueError(f"Model type '{model_type}' not recognized.") 