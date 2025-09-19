from __future__ import annotations
from typing import Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from dr_former.embedding import TimeEmbedding, InvertedEmbedding, HorizonEmbedding, PredToFeatureTokens
from dr_former.positional import LearnedPositionalEncoding
from dr_former.fusion import MultiHeadCrossAttention
from dr_former.output_head import SwitchableResidualHead

ModeT = Literal["time_first", "inverted"]


class MultiHeadSelfAttention(nn.Module):
    """Self-attention wrapper using PyTorch's MultiheadAttention."""
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        out, _ = self.mha(x, x, x, need_weights=False)  # self-attention: Q=K=V=x
        return out


class FeedForwardNetwork(nn.Module):
    """Position-wise FFN with configurable activation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
        # Same activation logic as EncoderLayer
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        elif activation == "elu":
            self.activation = F.elu
        else:
            self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        y = self.activation(self.conv1(x.transpose(-1, 1)))  # [B, d_ff, L]
        y = self.dropout(y)  # Single dropout after activation
        y = self.conv2(y).transpose(-1, 1)  # [B, L, d_model]
        return y


class DualStreamBlock(nn.Module):
    """
    One interleaved fusion block:
    1. Parallel self-attention on both streams
    2. Bi-directional cross-attention (A ↔ B)
    3. Parallel FFNs on both streams
    All with pre-norm + residual connections.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        
        # Self-attention for both streams
        self.self_attn_a = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.self_attn_b = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm_sa_a = nn.LayerNorm(d_model)
        self.norm_sa_b = nn.LayerNorm(d_model)
        
        # Cross-attention (bi-directional)
        self.cross_attn_a = MultiHeadCrossAttention(d_model, num_heads, dropout)  # A queries B
        self.cross_attn_b = MultiHeadCrossAttention(d_model, num_heads, dropout)  # B queries A
        self.norm_ca_a = nn.LayerNorm(d_model)
        self.norm_ca_b = nn.LayerNorm(d_model)
        
        # FFNs for both streams
        self.ffn_a = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.ffn_b = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.norm_ffn_a = nn.LayerNorm(d_model)
        self.norm_ffn_b = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_a: Stream A [B, L_a, d_model] 
            z_b: Stream B [B, L_b, d_model]
        Returns:
            (z_a_out, z_b_out) with same shapes
        """
        
        # 1. Self-attention (pre-norm + residual)
        z_a = z_a + self.dropout(self.self_attn_a(self.norm_sa_a(z_a)))
        z_b = z_b + self.dropout(self.self_attn_b(self.norm_sa_b(z_b)))
        
        # 2. Cross-attention (bi-directional, pre-norm + residual)
        # A attends to B, B attends to A
        a2 = self.norm_ca_a(z_a); b2 = self.norm_ca_b(z_b)
        z_a = z_a + self.dropout(self.cross_attn_a(a2, b2, b2))
        z_b = z_b + self.dropout(self.cross_attn_b(b2, a2, a2))
        
        # 3. FFN (pre-norm + residual)
        z_a = z_a + self.dropout(self.ffn_a(self.norm_ffn_a(z_a)))
        z_b = z_b + self.dropout(self.ffn_b(self.norm_ffn_b(z_b)))
        
        return z_a, z_b


class SwitchableEmbedding(nn.Module):
    """Embedding that switches between time_first and inverted modes."""
    def __init__(self, mode: ModeT, seq_len: int, horizon: int, 
                num_features: int, out_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.mode = mode
        
        if mode == "time_first":
            self.embed_a = TimeEmbedding(num_features, d_model, dropout)     # [B,T,F] -> [B,T,d_model]
            self.embed_b = HorizonEmbedding(out_dim, d_model, dropout)       # [B,H,D] -> [B,H,d_model]
        else:  # inverted
            self.embed_a = InvertedEmbedding(seq_len, d_model, dropout)      # [B,T,F] -> [B,F,d_model] 
            self.embed_b = PredToFeatureTokens(horizon, out_dim, num_features, d_model, dropout)  # [B,H,D] -> [B,F,d_model]

    def forward(self, x: torch.Tensor, y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: sensors [B, T, F]
            y_pred: DR predictions [B, H, D]
        Returns:
            z_a, z_b embedded streams
        """
        z_a = self.embed_a(x)
        z_b = self.embed_b(y_pred)
        return z_a, z_b


class DualStreamTransformer(nn.Module):
    """
    Deep Interleaved Fusion Dual Encoder Transformer.
    
    Two streams processed in parallel with cross-attention fusion at each block.
    Stream A (sensors) and Stream B (DR predictions) exchange information 
    bidirectionally at every layer.
    """
    
    def __init__(self, *, mode: ModeT, num_blocks: int, seq_len: int, horizon: int,
                num_features: int, out_dim: int, d_model: int, num_heads: int, 
                d_ff: int, dropout: float, activation: str, target_indices: list = None):
        super().__init__()
        
        self.mode = mode
        self.horizon = horizon
        self.out_dim = out_dim
        
        # Embeddings
        self.embedding = SwitchableEmbedding(
            mode, seq_len, horizon, num_features, out_dim, d_model, dropout
        )
        
        # Shared positional encoding for time_first mode
        if mode == "time_first":
            self.pos_encoding = LearnedPositionalEncoding(d_model, max(seq_len, horizon))
        else:
            self.pos_encoding = None
            
        # Stack of dual-stream blocks
        self.blocks = nn.ModuleList([
            DualStreamBlock(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_blocks)
        ])
        
        # Output head
        self.output_head = SwitchableResidualHead(
            mode,
            T=seq_len, d_model=d_model,
            horizon=horizon, out_dim=out_dim,
            target_indices=target_indices
        )
    
    def forward(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: sensors [B, T, F]
            y_pred: DR predictions [B, H, D]
        Returns:
            residuals [B, H, D]
        """
        
        # 1. Embedding
        z_a, z_b = self.embedding(x, y_pred)
        
        # 2. Positional encoding (time_first only)
        if self.pos_encoding is not None:
            z_a = self.pos_encoding(z_a)
            z_b = self.pos_encoding(z_b)
        
        # 3. Process through dual-stream blocks
        for block in self.blocks:
            z_a, z_b = block(z_a, z_b)
        
        # 4. Output head (directly from final block)
        residuals = self.output_head(z_a)
            
        return residuals