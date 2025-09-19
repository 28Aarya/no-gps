<<<<<<< HEAD
import torch
import torch.nn as nn

class TrajectoryPredictionHead(nn.Module):
    def __init__(self, d_model: int, prediction_length: int = 20, target_dim: int = 3):
        super().__init__()
        
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        
        # Global average pooling + projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Pool over sequence dimension
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, prediction_length * target_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        batch_size = x.shape[0]
        
        # Global average pooling over sequence dimension
        pooled = self.global_pool(x.transpose(1, 2))  # (batch, d_model, 1)
        pooled = pooled.squeeze(-1)  # (batch, d_model)
        
        # Project to output space
        output = self.projection(pooled)  # (batch, prediction_length * target_dim)
        output = output.view(batch_size, self.prediction_length, self.target_dim)
        
        return output
=======
import torch
import torch.nn as nn

class TrajectoryPredictionHead(nn.Module):
    def __init__(self, d_model: int, prediction_length: int = 20, target_dim: int = 3):
        super().__init__()
        
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        
        # Global average pooling + projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Pool over sequence dimension
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, prediction_length * target_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        batch_size = x.shape[0]
        
        # Global average pooling over sequence dimension
        pooled = self.global_pool(x.transpose(1, 2))  # (batch, d_model, 1)
        pooled = pooled.squeeze(-1)  # (batch, d_model)
        
        # Project to output space
        output = self.projection(pooled)  # (batch, prediction_length * target_dim)
        output = output.view(batch_size, self.prediction_length, self.target_dim)
        
        return output
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
