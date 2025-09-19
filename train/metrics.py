<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction"""
    def __init__(self, loss_type: str = 'mse', alpha: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (batch, seq_len, 3) - lat, lon, altitude
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred, target)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(pred, target)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss

class TrajectoryMetrics:
    """Evaluation metrics for trajectory prediction"""
    
    @staticmethod
    def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate various trajectory metrics"""
        with torch.no_grad():
            # MSE
            mse = F.mse_loss(pred, target).item()
            
            # MAE
            mae = F.l1_loss(pred, target).item()
            
            # RMSE
            rmse = torch.sqrt(F.mse_loss(pred, target)).item()
            
            # Position error (Euclidean distance for lat/lon)
            pos_pred = pred[:, :, :2]  # lat, lon
            pos_target = target[:, :, :2]
            pos_error = torch.sqrt(torch.sum((pos_pred - pos_target) ** 2, dim=-1))
            mean_pos_error = pos_error.mean().item()
            
            # Altitude error
            alt_pred = pred[:, :, 2]  # altitude
            alt_target = target[:, :, 2]
            alt_error = torch.abs(alt_pred - alt_target).mean().item()
            
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'position_error': mean_pos_error,
            'altitude_error': alt_error
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction"""
    def __init__(self, loss_type: str = 'mse', alpha: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (batch, seq_len, 3) - lat, lon, altitude
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred, target)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(pred, target)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss

class TrajectoryMetrics:
    """Evaluation metrics for trajectory prediction"""
    
    @staticmethod
    def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate various trajectory metrics"""
        with torch.no_grad():
            # MSE
            mse = F.mse_loss(pred, target).item()
            
            # MAE
            mae = F.l1_loss(pred, target).item()
            
            # RMSE
            rmse = torch.sqrt(F.mse_loss(pred, target)).item()
            
            # Position error (Euclidean distance for lat/lon)
            pos_pred = pred[:, :, :2]  # lat, lon
            pos_target = target[:, :, :2]
            pos_error = torch.sqrt(torch.sum((pos_pred - pos_target) ** 2, dim=-1))
            mean_pos_error = pos_error.mean().item()
            
            # Altitude error
            alt_pred = pred[:, :, 2]  # altitude
            alt_target = target[:, :, 2]
            alt_error = torch.abs(alt_pred - alt_target).mean().item()
            
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'position_error': mean_pos_error,
            'altitude_error': alt_error
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
        }