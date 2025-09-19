<<<<<<< HEAD
import torch
from torch import nn
from .patch_embedding import PatchEmbedding
from .output_head import OutputHead, CrossChannelFusion
from .encoder_wrapper import EncoderWrapper


class PatchTST(nn.Module):
    """
    Full PatchTST model combining embedding, encoder, and output head.
    """
    def __init__(self, config, encoder_stack):
        super().__init__()
        self.embedding = PatchEmbedding(config.patch_len, config.d_model)
        self.encoder_stack = encoder_stack  # vanilla encoder stack wrapper that accepts (B, C, P, D)
        if getattr(config, 'fusion', False):
            self.output_head = CrossChannelFusion(
                num_channels=getattr(config, 'num_channels', None) or 0,
                d_model=config.d_model,
                pred_len=config.pred_len,
                output_dim=config.output_dim
            )
        else:
            self.output_head = OutputHead(
                d_model=config.d_model,
                pred_len=config.pred_len,
                output_dim=config.output_dim
            )

    def forward(self, x):
        # x expected: (batch, num_channels, num_patches, patch_len)
        x = self.embedding(x)
        x = self.encoder_stack(x)
        x = self.output_head(x)
        return x


def build_patchtst_model(config) -> PatchTST:
    """
    Build PatchTST with encoder-only stack wrapped for channel-independent attention.
    Expects config.model to contain: patch_len, d_model, n_heads, e_layers, d_ff, dropout,
    pred_len, output_dim, and optional fusion and debug_shapes.
    """
    encoder = EncoderWrapper(
        encoder_layer_class=None,
        num_layers=config.e_layers,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        debug_shapes=getattr(config, 'debug_shapes', False),
    )
    return PatchTST(config, encoder_stack=encoder)
=======
import torch
from torch import nn
from .patch_embedding import PatchEmbedding
from .output_head import OutputHead, CrossChannelFusion
from .encoder_wrapper import EncoderWrapper


class PatchTST(nn.Module):
    """
    Full PatchTST model combining embedding, encoder, and output head.
    """
    def __init__(self, config, encoder_stack):
        super().__init__()
        self.embedding = PatchEmbedding(config.patch_len, config.d_model)
        self.encoder_stack = encoder_stack  # vanilla encoder stack wrapper that accepts (B, C, P, D)
        if getattr(config, 'fusion', False):
            self.output_head = CrossChannelFusion(
                num_channels=getattr(config, 'num_channels', None) or 0,
                d_model=config.d_model,
                pred_len=config.pred_len,
                output_dim=config.output_dim
            )
        else:
            self.output_head = OutputHead(
                d_model=config.d_model,
                pred_len=config.pred_len,
                output_dim=config.output_dim
            )

    def forward(self, x):
        # x expected: (batch, num_channels, num_patches, patch_len)
        x = self.embedding(x)
        x = self.encoder_stack(x)
        x = self.output_head(x)
        return x


def build_patchtst_model(config) -> PatchTST:
    """
    Build PatchTST with encoder-only stack wrapped for channel-independent attention.
    Expects config.model to contain: patch_len, d_model, n_heads, e_layers, d_ff, dropout,
    pred_len, output_dim, and optional fusion and debug_shapes.
    """
    encoder = EncoderWrapper(
        encoder_layer_class=None,
        num_layers=config.e_layers,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        debug_shapes=getattr(config, 'debug_shapes', False),
    )
    return PatchTST(config, encoder_stack=encoder)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
