<<<<<<< HEAD
import torch
from torch import nn


class ChannelAttention(nn.Module):
    """
    Channel-independent attention per patch.

    Applies self-attention across the `num_patches` dimension independently for
    each channel, preserving the shape (B, C, P, D).
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: Tensor of shape (batch, num_channels, num_patches, d_model)
        Returns: same shape
        """
        if x.dim() != 4:
            raise ValueError(f"ChannelAttention expects 4D input (B, C, P, D). Got shape: {tuple(x.shape)}")

        batch_size, num_channels, num_patches, d_model = x.shape
        x_reshaped = x.reshape(batch_size * num_channels, num_patches, d_model)
        attn_output, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped, need_weights=False)
        x_reshaped = self.norm(x_reshaped + self.dropout(attn_output))
        return x_reshaped.reshape(batch_size, num_channels, num_patches, d_model)
=======
import torch
from torch import nn


class ChannelAttention(nn.Module):
    """
    Channel-independent attention per patch.

    Applies self-attention across the `num_patches` dimension independently for
    each channel, preserving the shape (B, C, P, D).
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: Tensor of shape (batch, num_channels, num_patches, d_model)
        Returns: same shape
        """
        if x.dim() != 4:
            raise ValueError(f"ChannelAttention expects 4D input (B, C, P, D). Got shape: {tuple(x.shape)}")

        batch_size, num_channels, num_patches, d_model = x.shape
        x_reshaped = x.reshape(batch_size * num_channels, num_patches, d_model)
        attn_output, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped, need_weights=False)
        x_reshaped = self.norm(x_reshaped + self.dropout(attn_output))
        return x_reshaped.reshape(batch_size, num_channels, num_patches, d_model)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
