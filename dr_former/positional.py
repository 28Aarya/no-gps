<<<<<<< HEAD
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    # Only used in encoder-only 
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    def forward(self, x):  # [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T, :]
=======
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    # Only used in encoder-only 
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    def forward(self, x):  # [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T, :]
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
