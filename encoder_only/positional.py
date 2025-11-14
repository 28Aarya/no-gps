import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LearnedPositionalEncoding(nn.Module):
    """Learnable positional embeddings for aircraft trajectories"""
    def __init__(self, d_model: int, max_seq_length: int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pe = nn.Parameter(torch.randn(1, max_seq_length, d_model) * 0.02)  # Small init
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        if seq_len > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length {self.max_seq_length}")
        
        # Add positional encoding: (1, seq_len, d_model) + (batch, seq_len, d_model)
        return x + self.pe[:, :seq_len, :]