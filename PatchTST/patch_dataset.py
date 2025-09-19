<<<<<<< HEAD
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Any, Union


class PatchTSTDataset(Dataset):
    """
    On-the-fly PatchTST dataset that constructs patch sequences per flight (group).

    For each group (flight), we create rolling windows of length `seq_len + pred_len`.
    - The first `seq_len` timesteps become the model input X, split into patches of
    size `patch_len` with stride `patch_stride` to produce shape (C, P, patch_len).
    - The next `pred_len` timesteps become the target Y of shape (pred_len, O).

    Returns per item:
    X: Tensor (C, P, patch_len)  [float32]
    Y: Tensor (pred_len, O)      [float32]

    Notes:
    - All processing is done strictly within each `group_id` (no mixing).
    - P is fixed given seq_len, patch_len, patch_stride: P = floor((seq_len - patch_len)/patch_stride) + 1
    - Window step defaults to `window_stride` (if None, uses `patch_stride`).
    - Ensures dtype=float32 to be compatible with downstream Linear/Projection layers.
    """

    def __init__(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        group_ids: Optional[List[Any]] = None,
        *,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        patch_stride: Optional[int] = None,
        window_stride: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)

        if x.dim() != 3:
            raise ValueError(f"x expected shape (N, T, C). Got: {tuple(x.shape)}")
        if y.dim() != 3:
            raise ValueError(f"y expected shape (N, T, O). Got: {tuple(y.shape)}")
        if x.shape[:2] != y.shape[:2]:
            raise ValueError("x and y must share (N, T)")

        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available. Set device explicitly if you want CPU.")
            preferred_device = torch.device('cuda')
        else:
            preferred_device = device

        self.x = x.to(dtype=torch.float32, device=preferred_device)
        self.y = y.to(dtype=torch.float32, device=preferred_device)
        
        self.group_ids = list(group_ids) if group_ids is not None else list(range(self.x.shape[0]))
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride) if patch_stride is not None else int(patch_len)
        self.window_stride = int(window_stride) if window_stride is not None else int(self.patch_stride)

        if self.patch_len <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be positive")
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")

        # device placement preference
        self.device = preferred_device

        # Precompute group → indices mapping (preserving order)
        gid_to_indices: Dict[Any, List[int]] = {}
        for idx, gid in enumerate(self.group_ids):
            gid_to_indices.setdefault(gid, []).append(idx)

        self.groups: List[Tuple[Any, torch.Tensor]] = []
        for gid, idxs in gid_to_indices.items():
            if len(idxs) == 0:
                continue
            idxs_t = torch.as_tensor(idxs, device=self.x.device)
            # Flatten time along group
            xg = self.x.index_select(0, idxs_t)  # (n_g, T, C)
            yg = self.y.index_select(0, idxs_t)  # (n_g, T, O)
            xg = xg.reshape(-1, xg.shape[-1])    # (T_total, C)
            yg = yg.reshape(-1, yg.shape[-1])    # (T_total, O)

            if xg.shape[0] >= (self.seq_len + self.pred_len):
                self.groups.append((gid, idxs_t))

        # Build a global index: list of (group_idx, start_time)
        self.sample_index: List[Tuple[int, int]] = []
        self.P: Optional[int] = None
        for g_idx, (_, idxs_t) in enumerate(self.groups):
            xg = self.x.index_select(0, idxs_t).reshape(-1, self.x.shape[-1])  # (T_total, C)
            T_total = xg.shape[0]
            # Number of windows for this group
            max_start = T_total - (self.seq_len + self.pred_len)
            if max_start < 0:
                continue
            starts = range(0, max_start + 1, self.window_stride)
            # Compute P once using seq_len/patch params
            if self.P is None:
                # Unfold on a dummy tensor to infer P deterministically
                dummy = torch.empty((self.seq_len,), dtype=torch.float32)
                P = dummy.unfold(0, size=self.patch_len, step=self.patch_stride).shape[0]
                if P <= 0:
                    raise ValueError("Invalid (seq_len, patch_len, patch_stride) produce zero patches")
                self.P = int(P)
            for s in starts:
                self.sample_index.append((g_idx, int(s)))

        if self.P is None:
            # No valid samples
            self.P = 0

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g_idx, start = self.sample_index[idx]
        gid, idxs_t = self.groups[g_idx]

        # Fetch flattened group tensors
        xg = self.x.index_select(0, idxs_t).reshape(-1, self.x.shape[-1])  # (T_total, C)
        yg = self.y.index_select(0, idxs_t).reshape(-1, self.y.shape[-1])  # (T_total, O)

        # Window slice
        x_win = xg[start : start + self.seq_len]             # (seq_len, C)
        y_win = yg[start + self.seq_len : start + self.seq_len + self.pred_len]  # (pred_len, O)

        # Build patches from x_win: (C, P, patch_len)
        x_ct = x_win.transpose(0, 1)                         # (C, seq_len)
        x_unf = x_ct.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)  # (C, P, patch_len)

        # Cast and device placement
        X = x_unf.to(dtype=torch.float32, device=self.device).contiguous()
        Y = y_win.to(dtype=torch.float32, device=self.device).contiguous()

        return X, Y

    # ---- Debug helpers ----
    def get_sample_components(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (x_win, X_patches, y_win) for inspection.
        Shapes: x_win (seq_len, C), X_patches (C, P, patch_len), y_win (pred_len, O)
        """
        g_idx, start = self.sample_index[idx]
        _, idxs_t = self.groups[g_idx]
        xg = self.x.index_select(0, idxs_t).reshape(-1, self.x.shape[-1])
        yg = self.y.index_select(0, idxs_t).reshape(-1, self.y.shape[-1])
        x_win = xg[start : start + self.seq_len]
        y_win = yg[start + self.seq_len : start + self.seq_len + self.pred_len]
        x_ct = x_win.transpose(0, 1)
        X_patches = x_ct.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        return x_win, X_patches, y_win

    def print_debug_example(self, idx: int = 0, channel_index: int = 0, max_patches: int = 3) -> None:
        """
        Prints raw series for one channel and its first few patches to verify contiguity and no cross-channel mixing.
        """
        x_win, X_patches, y_win = self.get_sample_components(idx)
        x_ch = x_win[:, channel_index]
        print(f"[DEBUG] x_win (seq_len={self.seq_len}) channel {channel_index}:\n{x_ch.detach().cpu().numpy()}")
        num_show = min(max_patches, X_patches.shape[1])
        for p in range(num_show):
            patch = X_patches[channel_index, p]
            print(f"[DEBUG] patch {p} (len={self.patch_len}) for channel {channel_index}:\n{patch.detach().cpu().numpy()}")


=======
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Any, Union


class PatchTSTDataset(Dataset):
    """
    On-the-fly PatchTST dataset that constructs patch sequences per flight (group).

    For each group (flight), we create rolling windows of length `seq_len + pred_len`.
    - The first `seq_len` timesteps become the model input X, split into patches of
    size `patch_len` with stride `patch_stride` to produce shape (C, P, patch_len).
    - The next `pred_len` timesteps become the target Y of shape (pred_len, O).

    Returns per item:
    X: Tensor (C, P, patch_len)  [float32]
    Y: Tensor (pred_len, O)      [float32]

    Notes:
    - All processing is done strictly within each `group_id` (no mixing).
    - P is fixed given seq_len, patch_len, patch_stride: P = floor((seq_len - patch_len)/patch_stride) + 1
    - Window step defaults to `window_stride` (if None, uses `patch_stride`).
    - Ensures dtype=float32 to be compatible with downstream Linear/Projection layers.
    """

    def __init__(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        group_ids: Optional[List[Any]] = None,
        *,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        patch_stride: Optional[int] = None,
        window_stride: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)

        if x.dim() != 3:
            raise ValueError(f"x expected shape (N, T, C). Got: {tuple(x.shape)}")
        if y.dim() != 3:
            raise ValueError(f"y expected shape (N, T, O). Got: {tuple(y.shape)}")
        if x.shape[:2] != y.shape[:2]:
            raise ValueError("x and y must share (N, T)")

        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available. Set device explicitly if you want CPU.")
            preferred_device = torch.device('cuda')
        else:
            preferred_device = device

        self.x = x.to(dtype=torch.float32, device=preferred_device)
        self.y = y.to(dtype=torch.float32, device=preferred_device)
        
        self.group_ids = list(group_ids) if group_ids is not None else list(range(self.x.shape[0]))
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride) if patch_stride is not None else int(patch_len)
        self.window_stride = int(window_stride) if window_stride is not None else int(self.patch_stride)

        if self.patch_len <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be positive")
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")

        # device placement preference
        self.device = preferred_device

        # Precompute group → indices mapping (preserving order)
        gid_to_indices: Dict[Any, List[int]] = {}
        for idx, gid in enumerate(self.group_ids):
            gid_to_indices.setdefault(gid, []).append(idx)

        self.groups: List[Tuple[Any, torch.Tensor]] = []
        for gid, idxs in gid_to_indices.items():
            if len(idxs) == 0:
                continue
            idxs_t = torch.as_tensor(idxs, device=self.x.device)
            # Flatten time along group
            xg = self.x.index_select(0, idxs_t)  # (n_g, T, C)
            yg = self.y.index_select(0, idxs_t)  # (n_g, T, O)
            xg = xg.reshape(-1, xg.shape[-1])    # (T_total, C)
            yg = yg.reshape(-1, yg.shape[-1])    # (T_total, O)

            if xg.shape[0] >= (self.seq_len + self.pred_len):
                self.groups.append((gid, idxs_t))

        # Build a global index: list of (group_idx, start_time)
        self.sample_index: List[Tuple[int, int]] = []
        self.P: Optional[int] = None
        for g_idx, (_, idxs_t) in enumerate(self.groups):
            xg = self.x.index_select(0, idxs_t).reshape(-1, self.x.shape[-1])  # (T_total, C)
            T_total = xg.shape[0]
            # Number of windows for this group
            max_start = T_total - (self.seq_len + self.pred_len)
            if max_start < 0:
                continue
            starts = range(0, max_start + 1, self.window_stride)
            # Compute P once using seq_len/patch params
            if self.P is None:
                # Unfold on a dummy tensor to infer P deterministically
                dummy = torch.empty((self.seq_len,), dtype=torch.float32)
                P = dummy.unfold(0, size=self.patch_len, step=self.patch_stride).shape[0]
                if P <= 0:
                    raise ValueError("Invalid (seq_len, patch_len, patch_stride) produce zero patches")
                self.P = int(P)
            for s in starts:
                self.sample_index.append((g_idx, int(s)))

        if self.P is None:
            # No valid samples
            self.P = 0

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g_idx, start = self.sample_index[idx]
        gid, idxs_t = self.groups[g_idx]

        # Fetch flattened group tensors
        xg = self.x.index_select(0, idxs_t).reshape(-1, self.x.shape[-1])  # (T_total, C)
        yg = self.y.index_select(0, idxs_t).reshape(-1, self.y.shape[-1])  # (T_total, O)

        # Window slice
        x_win = xg[start : start + self.seq_len]             # (seq_len, C)
        y_win = yg[start + self.seq_len : start + self.seq_len + self.pred_len]  # (pred_len, O)

        # Build patches from x_win: (C, P, patch_len)
        x_ct = x_win.transpose(0, 1)                         # (C, seq_len)
        x_unf = x_ct.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)  # (C, P, patch_len)

        # Cast and device placement
        X = x_unf.to(dtype=torch.float32, device=self.device).contiguous()
        Y = y_win.to(dtype=torch.float32, device=self.device).contiguous()

        return X, Y

    # ---- Debug helpers ----
    def get_sample_components(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (x_win, X_patches, y_win) for inspection.
        Shapes: x_win (seq_len, C), X_patches (C, P, patch_len), y_win (pred_len, O)
        """
        g_idx, start = self.sample_index[idx]
        _, idxs_t = self.groups[g_idx]
        xg = self.x.index_select(0, idxs_t).reshape(-1, self.x.shape[-1])
        yg = self.y.index_select(0, idxs_t).reshape(-1, self.y.shape[-1])
        x_win = xg[start : start + self.seq_len]
        y_win = yg[start + self.seq_len : start + self.seq_len + self.pred_len]
        x_ct = x_win.transpose(0, 1)
        X_patches = x_ct.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        return x_win, X_patches, y_win

    def print_debug_example(self, idx: int = 0, channel_index: int = 0, max_patches: int = 3) -> None:
        """
        Prints raw series for one channel and its first few patches to verify contiguity and no cross-channel mixing.
        """
        x_win, X_patches, y_win = self.get_sample_components(idx)
        x_ch = x_win[:, channel_index]
        print(f"[DEBUG] x_win (seq_len={self.seq_len}) channel {channel_index}:\n{x_ch.detach().cpu().numpy()}")
        num_show = min(max_patches, X_patches.shape[1])
        for p in range(num_show):
            patch = X_patches[channel_index, p]
            print(f"[DEBUG] patch {p} (len={self.patch_len}) for channel {channel_index}:\n{patch.detach().cpu().numpy()}")


>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
