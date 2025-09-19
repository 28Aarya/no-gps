<<<<<<< HEAD
import torch
import torch.nn as nn
from .encoder_layer import EncoderLayer

class EncoderStack(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        
        for layer in self.layers:
            x = layer(x, mask)
            
=======
import torch
import torch.nn as nn
from .encoder_layer import EncoderLayer

class EncoderStack(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        
        for layer in self.layers:
            x = layer(x, mask)
            
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
        return x