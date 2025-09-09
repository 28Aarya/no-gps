<<<<<<< HEAD
import torch.nn as nn
from iTransformer.layers.Transformer_EncDec import Encoder, EncoderLayer
from iTransformer.layers.SelfAttention_Family import FullAttention, AttentionLayer

def build_encoder_stack(d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1, activation: str = "gelu"):
    layers = [
        EncoderLayer(
            AttentionLayer(
                FullAttention(mask_flag=False, factor=5, attention_dropout=dropout, output_attention=False),
                d_model=d_model, n_heads=n_heads
            ),
            d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation
        )
        for _ in range(n_layers)
    ]
    return Encoder(layers, norm_layer=nn.LayerNorm(d_model))
=======
import torch.nn as nn
from iTransformer.layers.Transformer_EncDec import Encoder, EncoderLayer
from iTransformer.layers.SelfAttention_Family import FullAttention, AttentionLayer

def build_encoder_stack(d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1, activation: str = "gelu"):
    layers = [
        EncoderLayer(
            AttentionLayer(
                FullAttention(mask_flag=False, factor=5, attention_dropout=dropout, output_attention=False),
                d_model=d_model, n_heads=n_heads
            ),
            d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation
        )
        for _ in range(n_layers)
    ]
    return Encoder(layers, norm_layer=nn.LayerNorm(d_model))
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
