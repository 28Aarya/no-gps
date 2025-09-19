<<<<<<< HEAD
import torch
from torch import nn


class OutputHead(nn.Module):
    """
    Projects encoder output to prediction dimension, channel-independent.
    """
    def __init__(self, d_model, pred_len, output_dim):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.proj = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, x):
        """
        x: Tensor (batch, num_channels, num_patches, d_model)
        Returns: Tensor (batch, pred_len, output_dim)
        """
        if x.dim() != 4:
            raise ValueError(f"OutputHead expects 4D input (B, C, P, D). Got shape: {tuple(x.shape)}")
        # Global mean pool across channels and patches
        x_pooled = x.mean(dim=(1, 2))  # (B, D)
        y = self.proj(x_pooled)  # (B, pred_len * output_dim)
        return y.view(x.shape[0], self.pred_len, self.output_dim)


class CrossChannelFusion(nn.Module):
    """
    Optional linear fusion across channels.
    """
    def __init__(self, num_channels, d_model, pred_len, output_dim):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        # Use provided num_channels if > 0; otherwise initialize lazily
        if isinstance(num_channels, int) and num_channels > 0:
            self.channel_fuse = nn.Linear(num_channels, 1, bias=False)
        else:
            self.channel_fuse = nn.LazyLinear(1, bias=False)
        self.proj = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, x):
        """
        x: Tensor (batch, num_channels, num_patches, d_model)
        Returns: Tensor (batch, pred_len, output_dim)
        """
        if x.dim() != 4:
            raise ValueError(f"CrossChannelFusion expects 4D input (B, C, P, D). Got shape: {tuple(x.shape)}")
        # First pool patches per channel
        x_channel = x.mean(dim=2)  # (B, C, D)
        # Fuse channels across C dimension using a linear layer applied on last dim after transpose
        x_fused = self.channel_fuse(x_channel.transpose(1, 2)).transpose(1, 2).squeeze(1)  # (B, D)
        y = self.proj(x_fused)
        return y.view(x.shape[0], self.pred_len, self.output_dim)
=======
import torch
from torch import nn


class OutputHead(nn.Module):
    """
    Projects encoder output to prediction dimension, channel-independent.
    """
    def __init__(self, d_model, pred_len, output_dim):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.proj = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, x):
        """
        x: Tensor (batch, num_channels, num_patches, d_model)
        Returns: Tensor (batch, pred_len, output_dim)
        """
        if x.dim() != 4:
            raise ValueError(f"OutputHead expects 4D input (B, C, P, D). Got shape: {tuple(x.shape)}")
        # Global mean pool across channels and patches
        x_pooled = x.mean(dim=(1, 2))  # (B, D)
        y = self.proj(x_pooled)  # (B, pred_len * output_dim)
        return y.view(x.shape[0], self.pred_len, self.output_dim)


class CrossChannelFusion(nn.Module):
    """
    Optional linear fusion across channels.
    """
    def __init__(self, num_channels, d_model, pred_len, output_dim):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        # Use provided num_channels if > 0; otherwise initialize lazily
        if isinstance(num_channels, int) and num_channels > 0:
            self.channel_fuse = nn.Linear(num_channels, 1, bias=False)
        else:
            self.channel_fuse = nn.LazyLinear(1, bias=False)
        self.proj = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, x):
        """
        x: Tensor (batch, num_channels, num_patches, d_model)
        Returns: Tensor (batch, pred_len, output_dim)
        """
        if x.dim() != 4:
            raise ValueError(f"CrossChannelFusion expects 4D input (B, C, P, D). Got shape: {tuple(x.shape)}")
        # First pool patches per channel
        x_channel = x.mean(dim=2)  # (B, C, D)
        # Fuse channels across C dimension using a linear layer applied on last dim after transpose
        x_fused = self.channel_fuse(x_channel.transpose(1, 2)).transpose(1, 2).squeeze(1)  # (B, D)
        y = self.proj(x_fused)
        return y.view(x.shape[0], self.pred_len, self.output_dim)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
