<<<<<<< HEAD
import torch
import torch.nn as nn

class ResidualHeadTimeTokens(nn.Module):
    # For encoder-only mode: input tokens along horizon [B, H, d_model] -> [B, H, D]
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)
    def forward(self, x):  # [B, H, d_model]
        return self.proj(x)

class ResidualHeadInverted(nn.Module):
    # For inverted mode: tokens = features [B, F, d_model]
    # Select target feature indices, project each to horizon, then stack -> [B, H, D]
    def __init__(self, d_model: int, horizon: int, target_indices):
        super().__init__()
        self.horizon = horizon
        self.target_indices = list(target_indices)
        self.head = nn.Linear(d_model, horizon)
    def forward(self, x):  # x: [B, F, d_model]
        sel = x[:, self.target_indices, :]                 # [B, D, d_model]
        H = self.head(sel)                                 # [B, D, H]
        return H.transpose(1, 2)                           # [B, H, D]

class ResidualHeadTimeFlatten(nn.Module):
    # Always read from Stream A (sensors) in time_first:
    # [B, T, d_model] -> flatten -> [B, H, D]
    def __init__(self, T: int, d_model: int, horizon: int, out_dim: int):
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.proj = nn.Linear(T * d_model, horizon * out_dim)
    def forward(self, x):  # x: [B, T, d_model] from Stream A
        B = x.size(0)
        y = self.proj(x.reshape(B, -1))                  # [B, H*D]
        return y.view(B, self.horizon, self.out_dim)     # [B, H, D]

class SwitchableResidualHead(nn.Module):
    # Picks the correct head:
    # - time_first: ResidualHeadTimeFlatten on Stream A (T->H mapping)
    # - inverted:   ResidualHeadInverted on Stream A (F->H via selected features)
    def __init__(self, mode: str, *, T: int, d_model: int, horizon: int, out_dim: int, target_indices=None):
        super().__init__()
        if mode == "time_first":
            self.head = ResidualHeadTimeFlatten(T, d_model, horizon, out_dim)
        else:
            if target_indices is None:
                target_indices = [0, 1, 2]
            self.head = ResidualHeadInverted(d_model, horizon, target_indices)
    def forward(self, x):
        return self.head(x)

=======
import torch
import torch.nn as nn

class ResidualHeadTimeTokens(nn.Module):
    # For encoder-only mode: input tokens along horizon [B, H, d_model] -> [B, H, D]
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)
    def forward(self, x):  # [B, H, d_model]
        return self.proj(x)

class ResidualHeadInverted(nn.Module):
    # For inverted mode: tokens = features [B, F, d_model]
    # Select target feature indices, project each to horizon, then stack -> [B, H, D]
    def __init__(self, d_model: int, horizon: int, target_indices):
        super().__init__()
        self.horizon = horizon
        self.target_indices = list(target_indices)
        self.head = nn.Linear(d_model, horizon)
    def forward(self, x):  # x: [B, F, d_model]
        sel = x[:, self.target_indices, :]                 # [B, D, d_model]
        H = self.head(sel)                                 # [B, D, H]
        return H.transpose(1, 2)                           # [B, H, D]

class ResidualHeadTimeFlatten(nn.Module):
    # Always read from Stream A (sensors) in time_first:
    # [B, T, d_model] -> flatten -> [B, H, D]
    def __init__(self, T: int, d_model: int, horizon: int, out_dim: int):
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.proj = nn.Linear(T * d_model, horizon * out_dim)
    def forward(self, x):  # x: [B, T, d_model] from Stream A
        B = x.size(0)
        y = self.proj(x.reshape(B, -1))                  # [B, H*D]
        return y.view(B, self.horizon, self.out_dim)     # [B, H, D]

class SwitchableResidualHead(nn.Module):
    # Picks the correct head:
    # - time_first: ResidualHeadTimeFlatten on Stream A (T->H mapping)
    # - inverted:   ResidualHeadInverted on Stream A (F->H via selected features)
    def __init__(self, mode: str, *, T: int, d_model: int, horizon: int, out_dim: int, target_indices=None):
        super().__init__()
        if mode == "time_first":
            self.head = ResidualHeadTimeFlatten(T, d_model, horizon, out_dim)
        else:
            if target_indices is None:
                target_indices = [0, 1, 2]
            self.head = ResidualHeadInverted(d_model, horizon, target_indices)
    def forward(self, x):
        return self.head(x)

>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
