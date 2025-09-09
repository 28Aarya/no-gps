<<<<<<< HEAD
import torch.nn as nn
from .embedding import TimeEmbedding, InvertedEmbedding, HorizonEmbedding, PredToFeatureTokens
from .positional import LearnedPositionalEncoding
from .fusion import CrossAttentionFusionScratch
from .encoder_stack import build_encoder_stack
from .output_head import ResidualHeadTimeTokens, ResidualHeadInverted

class DRResidualFormer(nn.Module):
    """
    Two-stream residual model. Inputs:
    - x: [B, T, F]        (past window features)
    - y_pred: [B, H, D]   (DR predictions)
    Outputs:
    - y_res: [B, H, D]    (residuals to add to y_pred)
    mode: 'encoder' -> time-as-tokens; 'inverted' -> features-as-tokens
    """
    def __init__(self, *, mode: str,
                seq_len: int, horizon: int, num_features: int, out_dim: int,
                d_model: int, n_heads: int, d_ff: int, n_layers: int,
                dropout: float = 0.1, max_len: int = 512, target_indices=None, activation: str = "gelu"):
        super().__init__()
        self.mode = mode
        self.horizon = horizon
        self.out_dim = out_dim

        if mode == "encoder":
            self.embed_x = TimeEmbedding(num_features, d_model, dropout)
            self.pos = LearnedPositionalEncoding(d_model, max_len)
            self.embed_pred = HorizonEmbedding(out_dim, d_model, dropout)
            self.fuse = CrossAttentionFusionScratch(d_model, n_heads, dropout)
            self.encoder = build_encoder_stack(d_model, n_heads, d_ff, n_layers, dropout, activation)
            self.head = ResidualHeadTimeTokens(d_model, out_dim)
            
        elif mode == "inverted":
            self.embed_x = InvertedEmbedding(seq_len, d_model, dropout)
            self.embed_pred = PredToFeatureTokens(horizon, out_dim, num_features, d_model, dropout)
            self.fuse = CrossAttentionFusionScratch(d_model, n_heads, dropout)
            self.encoder = build_encoder_stack(d_model, n_heads, d_ff, n_layers, dropout, activation)
            if target_indices is None:
                raise ValueError("target_indices (e.g., [7,8,9] for x,y,z) required for inverted mode")
            self.head = ResidualHeadInverted(d_model, horizon, target_indices)
        else:
            raise ValueError("mode must be 'encoder' or 'inverted'")

    def forward(self, x, y_pred):
        if self.mode == "encoder":
            ex = self.pos(self.embed_x(x))          # [B, T, d_model]
            ep = self.embed_pred(y_pred)            # [B, H, d_model]
            fused = self.fuse(ex, ep)               # queries=x tokens (sensors); keys/values=y_pred tokens (DR) -> [B, T, d_model]
            enc = self.encoder(fused)[0]            # [B, H, d_model]
            y_res = self.head(enc)                  # [B, H, D]

        else:
            ex = self.embed_x(x)                    # [B, F, d_model]
            ep = self.embed_pred(y_pred)            # [B, F, d_model]
            fused = self.fuse(ex, ep)               # queries=x tokens (sensors); keys/values=y_pred tokens (DR) -> [B, F, d_model]
            enc = self.encoder(fused)[0]            # [B, F, d_model]
            y_res = self.head(enc)                  # [B, H, D]
        return y_res
=======
import torch.nn as nn
from .embedding import TimeEmbedding, InvertedEmbedding, HorizonEmbedding, PredToFeatureTokens
from .positional import LearnedPositionalEncoding
from .fusion import CrossAttentionFusionScratch
from .encoder_stack import build_encoder_stack
from .output_head import ResidualHeadTimeTokens, ResidualHeadInverted

class DRResidualFormer(nn.Module):
    """
    Two-stream residual model. Inputs:
    - x: [B, T, F]        (past window features)
    - y_pred: [B, H, D]   (DR predictions)
    Outputs:
    - y_res: [B, H, D]    (residuals to add to y_pred)
    mode: 'encoder' -> time-as-tokens; 'inverted' -> features-as-tokens
    """
    def __init__(self, *, mode: str,
                seq_len: int, horizon: int, num_features: int, out_dim: int,
                d_model: int, n_heads: int, d_ff: int, n_layers: int,
                dropout: float = 0.1, max_len: int = 512, target_indices=None, activation: str = "gelu"):
        super().__init__()
        self.mode = mode
        self.horizon = horizon
        self.out_dim = out_dim

        if mode == "encoder":
            self.embed_x = TimeEmbedding(num_features, d_model, dropout)
            self.pos = LearnedPositionalEncoding(d_model, max_len)
            self.embed_pred = HorizonEmbedding(out_dim, d_model, dropout)
            self.fuse = CrossAttentionFusionScratch(d_model, n_heads, dropout)
            self.encoder = build_encoder_stack(d_model, n_heads, d_ff, n_layers, dropout, activation)
            self.head = ResidualHeadTimeTokens(d_model, out_dim)
            
        elif mode == "inverted":
            self.embed_x = InvertedEmbedding(seq_len, d_model, dropout)
            self.embed_pred = PredToFeatureTokens(horizon, out_dim, num_features, d_model, dropout)
            self.fuse = CrossAttentionFusionScratch(d_model, n_heads, dropout)
            self.encoder = build_encoder_stack(d_model, n_heads, d_ff, n_layers, dropout, activation)
            if target_indices is None:
                raise ValueError("target_indices (e.g., [7,8,9] for x,y,z) required for inverted mode")
            self.head = ResidualHeadInverted(d_model, horizon, target_indices)
        else:
            raise ValueError("mode must be 'encoder' or 'inverted'")

    def forward(self, x, y_pred):
        if self.mode == "encoder":
            ex = self.pos(self.embed_x(x))          # [B, T, d_model]
            ep = self.embed_pred(y_pred)            # [B, H, d_model]
            fused = self.fuse(ex, ep)               # queries=x tokens (sensors); keys/values=y_pred tokens (DR) -> [B, T, d_model]
            enc = self.encoder(fused)[0]            # [B, H, d_model]
            y_res = self.head(enc)                  # [B, H, D]

        else:
            ex = self.embed_x(x)                    # [B, F, d_model]
            ep = self.embed_pred(y_pred)            # [B, F, d_model]
            fused = self.fuse(ex, ep)               # queries=x tokens (sensors); keys/values=y_pred tokens (DR) -> [B, F, d_model]
            enc = self.encoder(fused)[0]            # [B, F, d_model]
            y_res = self.head(enc)                  # [B, H, D]
        return y_res
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
