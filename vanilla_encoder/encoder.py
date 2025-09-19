<<<<<<< HEAD
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
        
=======
import torch
import torch.nn as nn
from .positional import LearnedPositionalEncoding
from .encoder_stack import EncoderStack
from .heads import TrajectoryPredictionHead

class AircraftTrajectoryEncoder(nn.Module):
    def __init__(self, 
                input_features: int,
                icao_vocab_size: int,
                d_model: int = 64,
                num_heads: int = 8,
                num_layers: int = 6,
                d_ff: int = 256,
                dropout: float = 0.1,
                prediction_length: int = 20,
                target_dim: int = 3,
                max_seq_length: int = 100):
        super().__init__()
        
        self.input_features = input_features
        self.icao_vocab_size = icao_vocab_size
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        
        # Continuous feature projection (excluding ICAO ID)
        self.continuous_projection = nn.Linear(input_features - 1, d_model)
        
        # ICAO embedding
        self.icao_embedding = nn.Embedding(icao_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_length)
        
        # Encoder stack
        self.encoder_stack = EncoderStack(d_model, num_heads, num_layers, d_ff, dropout)
        
        # Output head
        self.output_head = TrajectoryPredictionHead(d_model, prediction_length, target_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_features) - last feature is ICAO ID
        batch_size, seq_len, _ = x.shape
        
        # Split continuous features and ICAO ID
        continuous_features = x[:, :, :-1]  # (batch, seq_len, input_features-1)
        icao_ids = x[:, :, -1].long()  # (batch, seq_len) - last feature as ICAO
        
        # Project continuous features
        continuous_proj = self.continuous_projection(continuous_features)  # (batch, seq_len, d_model)
        
        # Embed ICAO IDs
        icao_embedded = self.icao_embedding(icao_ids)  # (batch, seq_len, d_model)
        
        # Add continuous projection and ICAO embedding
        combined = continuous_proj + icao_embedded  # (batch, seq_len, d_model)
        
        # Add positional encoding
        encoded = self.pos_encoding(combined)  # (batch, seq_len, d_model)
        
        # Pass through encoder stack
        encoded = self.encoder_stack(encoded)  # (batch, seq_len, d_model)
        
        # Generate predictions
        predictions = self.output_head(encoded)  # (batch, prediction_length, target_dim)
        
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
        return predictions