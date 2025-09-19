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