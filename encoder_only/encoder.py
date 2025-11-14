import torch
import torch.nn as nn
from .positional import LearnedPositionalEncoding
from .encoder_stack import EncoderStack
from .heads import TrajectoryPredictionHead

class AircraftTrajectoryEncoder(nn.Module):
    def __init__(self, 
                input_features: int,
                d_model: int = 64,
                num_heads: int = 8,
                num_layers: int = 6,
                d_ff: int = 256,
                dropout: float = 0.1,
                prediction_length: int = 20,
                target_dim: int = 2,  
                max_seq_length: int = 100):
        super().__init__()
        
        self.input_features = input_features
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        
        # Feature projection (all features are continuous now)
        self.feature_projection = nn.Linear(input_features, d_model)
        
        # Positional encoding
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_length)
        
        # Encoder stack
        self.encoder_stack = EncoderStack(d_model, num_heads, num_layers, d_ff, dropout)
        
        # Output head
        self.output_head = TrajectoryPredictionHead(d_model, prediction_length, target_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_features) - all features are continuous
        batch_size, seq_len, _ = x.shape
        
        # Project all features
        projected = self.feature_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        encoded = self.pos_encoding(projected)  # (batch, seq_len, d_model)
        
        # Pass through encoder stack
        encoded = self.encoder_stack(encoded)  # (batch, seq_len, d_model)
        
        # Generate predictions
        predictions = self.output_head(encoded)  # (batch, prediction_length, target_dim)
        
        return predictions