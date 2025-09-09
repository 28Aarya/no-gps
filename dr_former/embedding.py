<<<<<<< HEAD
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    # [B, T, F] -> [B, T, d_model]
    def __init__(self, input_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_features, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):  # x: [B, T, F]
        return self.drop(self.proj(x))

class InvertedEmbedding(nn.Module):
    # [B, T, F] -> [B, F, d_model]  (features as tokens)
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):  # x: [B, T, F]
        x = x.permute(0, 2, 1)          # [B, F, T]
        return self.drop(self.proj(x))  # [B, F, d_model]

class HorizonEmbedding(nn.Module):
    # [B, H, D] -> [B, H, d_model] (for DR predictions stream in encoder-only mode)
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, y_pred):  # [B, H, D]
        return self.drop(self.proj(y_pred))

class PredToFeatureTokens(nn.Module):
    # [B, H, D] -> [B, F, d_model] (for inverted mode)
    # pool over H, then map D -> F tokens and project to d_model.
    def __init__(self, horizon: int, in_dim: int, num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_tokens = nn.Linear(in_dim, num_features)
        # Project each scalar token to a d_model-dimensional vector independently
        self.proj = nn.Linear(1, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, y_pred):  # [B, H, D]
        z = self.pool(y_pred.transpose(1, 2)).squeeze(-1)    # [B, D]
        tok = self.to_tokens(z).unsqueeze(-1)                # [B, F, 1]
        tok = self.proj(tok)                                 # [B, F, d_model]
        return self.drop(tok)
=======
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    # [B, T, F] -> [B, T, d_model]
    def __init__(self, input_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_features, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):  # x: [B, T, F]
        return self.drop(self.proj(x))

class InvertedEmbedding(nn.Module):
    # [B, T, F] -> [B, F, d_model]  (features as tokens)
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):  # x: [B, T, F]
        x = x.permute(0, 2, 1)          # [B, F, T]
        return self.drop(self.proj(x))  # [B, F, d_model]

class HorizonEmbedding(nn.Module):
    # [B, H, D] -> [B, H, d_model] (for DR predictions stream in encoder-only mode)
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, y_pred):  # [B, H, D]
        return self.drop(self.proj(y_pred))

class PredToFeatureTokens(nn.Module):
    # [B, H, D] -> [B, F, d_model] (for inverted mode)
    # pool over H, then map D -> F tokens and project to d_model.
    def __init__(self, horizon: int, in_dim: int, num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_tokens = nn.Linear(in_dim, num_features)
        # Project each scalar token to a d_model-dimensional vector independently
        self.proj = nn.Linear(1, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, y_pred):  # [B, H, D]
        z = self.pool(y_pred.transpose(1, 2)).squeeze(-1)    # [B, D]
        tok = self.to_tokens(z).unsqueeze(-1)                # [B, F, 1]
        tok = self.proj(tok)                                 # [B, F, d_model]
        return self.drop(tok)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
