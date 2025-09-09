<<<<<<< HEAD
import torch
import torch.nn as nn
import math

class MultiHeadCrossAttention(nn.Module):
    """
    Standard multi-head cross-attention:
    Q: [B, Lq, d_model], K,V: [B, Lk, d_model] -> out: [B, Lq, d_model]
    Stream-2 (y_pred) should be Q; Stream-1 (x) should be K,V.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        Q = self.W_q(q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)  # [B, h, Lq, d_k]
        K = self.W_k(k).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)  # [B, h, Lk, d_k]
        V = self.W_v(v).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)  # [B, h, Lk, d_k]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)    # [B, h, Lq, Lk]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        ctx = torch.matmul(attn, V)                                            # [B, h, Lq, d_k]
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lq, self.d_model)       # [B, Lq, d_model]
        out = self.W_o(ctx)                                                    # [B, Lq, d_model]
        return out

class CrossAttentionFusionScratch(nn.Module):
    """
    Thin wrapper: x tokens (sensors) as queries fuse with y_pred tokens (DR) as keys/values.
    Returns fused representation in the query's sequence length.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_residual: bool = True, use_norm: bool = True):
        super().__init__()
        self.mha = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.use_residual = use_residual
        self.norm = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        fused = self.mha(q, k, k, mask=None)           # [B, Lq, d_model]
        if self.use_residual:
            fused = q + self.drop(fused)
        return self.norm(fused)
=======
import torch
import torch.nn as nn
import math

class MultiHeadCrossAttention(nn.Module):
    """
    Standard multi-head cross-attention:
    Q: [B, Lq, d_model], K,V: [B, Lk, d_model] -> out: [B, Lq, d_model]
    Stream-2 (y_pred) should be Q; Stream-1 (x) should be K,V.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        Q = self.W_q(q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)  # [B, h, Lq, d_k]
        K = self.W_k(k).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)  # [B, h, Lk, d_k]
        V = self.W_v(v).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)  # [B, h, Lk, d_k]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)    # [B, h, Lq, Lk]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        ctx = torch.matmul(attn, V)                                            # [B, h, Lq, d_k]
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lq, self.d_model)       # [B, Lq, d_model]
        out = self.W_o(ctx)                                                    # [B, Lq, d_model]
        return out

class CrossAttentionFusionScratch(nn.Module):
    """
    Thin wrapper: x tokens (sensors) as queries fuse with y_pred tokens (DR) as keys/values.
    Returns fused representation in the query's sequence length.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_residual: bool = True, use_norm: bool = True):
        super().__init__()
        self.mha = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.use_residual = use_residual
        self.norm = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        fused = self.mha(q, k, k, mask=None)           # [B, Lq, d_model]
        if self.use_residual:
            fused = q + self.drop(fused)
        return self.norm(fused)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
