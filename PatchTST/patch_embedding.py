<<<<<<< HEAD
import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """
    Linear embedding for each patch.

    Maps the last dimension (patch samples) of size `patch_len` to `d_model`.
    """
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        """
        x: Tensor of shape (batch, num_channels, num_patches, patch_len)
        Returns: Tensor of shape (batch, num_channels, num_patches, d_model)
        """
        if x.dim() != 4:
            raise ValueError(f"PatchEmbedding expects 4D input (B, C, P, L). Got shape: {tuple(x.shape)}")
        return self.proj(x)
=======
import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """
    Linear embedding for each patch.

    Maps the last dimension (patch samples) of size `patch_len` to `d_model`.
    """
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        """
        x: Tensor of shape (batch, num_channels, num_patches, patch_len)
        Returns: Tensor of shape (batch, num_channels, num_patches, d_model)
        """
        if x.dim() != 4:
            raise ValueError(f"PatchEmbedding expects 4D input (B, C, P, L). Got shape: {tuple(x.shape)}")
        return self.proj(x)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
