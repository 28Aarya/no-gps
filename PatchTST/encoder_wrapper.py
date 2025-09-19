<<<<<<< HEAD
"""
Purpose: Wrap existing encoder_layer.py and encoder_stack.py from encoder_only/
to handle PatchTST input shapes (batch, num_channels, num_patches, d_model).

Classes/Functions:
- EncoderWrapper: adapts a single EncoderLayer to accept PatchTST patch inputs
                and stacks multiple layers using EncoderStack.
"""

import torch
from torch import nn
from encoder_only.encoder_stack import EncoderStack


class EncoderWrapper(nn.Module):
    """
    Wrapper to adapt vanilla encoder layers for PatchTST input.
    
    Parameters
    ----------
    encoder_layer_class : class
        The EncoderLayer class from encoder_only/encoder_layer.py
    num_layers : int
        Number of layers to stack
    d_model : int
        Embedding dimension of patches
    n_heads : int
        Number of attention heads
    d_ff : int
        Feed-forward dimension
    dropout : float
        Dropout probability
    
    Input shape
    -----------
    x: Tensor
        Shape: (batch, num_channels, num_patches, d_model)
    
    Output shape
    ------------
    x: Tensor
        Shape: (batch, num_channels, num_patches, d_model)
        Same shape as input, ready for output head
    """
    def __init__(self, encoder_layer_class, num_layers, d_model, n_heads, d_ff, dropout, debug_shapes: bool = False):
        super().__init__()
        # Note: encoder_layer_class is unused because EncoderStack constructs its own layers
        self.stack = EncoderStack(d_model=d_model, num_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
        self.debug_shapes = debug_shapes
    
    def forward(self, x):
        """
        Forward pass through stacked encoder layers.
        """
        if x.dim() != 4:
            raise ValueError(f"EncoderWrapper expects 4D input (B, C, P, D). Got shape: {tuple(x.shape)}")
        batch_size, num_channels, num_patches, d_model = x.shape
        if self.debug_shapes:
            print(f"[EncoderWrapper] input: B={batch_size}, C={num_channels}, P={num_patches}, D={d_model}")
        # Channel-independent processing: fold B and C together
        x_reshaped = x.reshape(batch_size * num_channels, num_patches, d_model)
        if self.debug_shapes:
            print(f"[EncoderWrapper] reshaped to: {(x_reshaped.shape,)} (treat channels independently)")
        x_encoded = self.stack(x_reshaped)
        out = x_encoded.reshape(batch_size, num_channels, num_patches, d_model)
        if self.debug_shapes:
            print(f"[EncoderWrapper] output: {out.shape}")
        return out
=======
"""
Purpose: Wrap existing encoder_layer.py and encoder_stack.py from encoder_only/
to handle PatchTST input shapes (batch, num_channels, num_patches, d_model).

Classes/Functions:
- EncoderWrapper: adapts a single EncoderLayer to accept PatchTST patch inputs
                and stacks multiple layers using EncoderStack.
"""

import torch
from torch import nn
from encoder_only.encoder_stack import EncoderStack


class EncoderWrapper(nn.Module):
    """
    Wrapper to adapt vanilla encoder layers for PatchTST input.
    
    Parameters
    ----------
    encoder_layer_class : class
        The EncoderLayer class from encoder_only/encoder_layer.py
    num_layers : int
        Number of layers to stack
    d_model : int
        Embedding dimension of patches
    n_heads : int
        Number of attention heads
    d_ff : int
        Feed-forward dimension
    dropout : float
        Dropout probability
    
    Input shape
    -----------
    x: Tensor
        Shape: (batch, num_channels, num_patches, d_model)
    
    Output shape
    ------------
    x: Tensor
        Shape: (batch, num_channels, num_patches, d_model)
        Same shape as input, ready for output head
    """
    def __init__(self, encoder_layer_class, num_layers, d_model, n_heads, d_ff, dropout, debug_shapes: bool = False):
        super().__init__()
        # Note: encoder_layer_class is unused because EncoderStack constructs its own layers
        self.stack = EncoderStack(d_model=d_model, num_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
        self.debug_shapes = debug_shapes
    
    def forward(self, x):
        """
        Forward pass through stacked encoder layers.
        """
        if x.dim() != 4:
            raise ValueError(f"EncoderWrapper expects 4D input (B, C, P, D). Got shape: {tuple(x.shape)}")
        batch_size, num_channels, num_patches, d_model = x.shape
        if self.debug_shapes:
            print(f"[EncoderWrapper] input: B={batch_size}, C={num_channels}, P={num_patches}, D={d_model}")
        # Channel-independent processing: fold B and C together
        x_reshaped = x.reshape(batch_size * num_channels, num_patches, d_model)
        if self.debug_shapes:
            print(f"[EncoderWrapper] reshaped to: {(x_reshaped.shape,)} (treat channels independently)")
        x_encoded = self.stack(x_reshaped)
        out = x_encoded.reshape(batch_size, num_channels, num_patches, d_model)
        if self.debug_shapes:
            print(f"[EncoderWrapper] output: {out.shape}")
        return out
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
