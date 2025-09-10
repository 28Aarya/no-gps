from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


def filter_outliers(Y_res, threshold=10000):
    """
    Filter out sequences with extremely large residuals.
    
    Args:
        Y_res: Residuals array [N, H, D]
        threshold: Maximum allowed residual magnitude in meters
        
    Returns:
        keep_mask: Boolean array [N] indicating which sequences to keep
    """
    abs_res = np.abs(Y_res)
    max_res = np.max(abs_res, axis=(1, 2))  # Max residual per sequence
    keep_mask = max_res < threshold
    return keep_mask


class DRResidualDataset(Dataset):
    def __init__(self, orig_npz_path: str, dr_pred_npz_path: str, dr_res_npz_path: str, norms: dict | None = None, outlier_threshold: float = 10000):
        if not os.path.exists(orig_npz_path):
            raise FileNotFoundError(orig_npz_path)
        if not os.path.exists(dr_pred_npz_path):
            raise FileNotFoundError(dr_pred_npz_path)
        if not os.path.exists(dr_res_npz_path):
            raise FileNotFoundError(dr_res_npz_path)

        d_orig = np.load(orig_npz_path, allow_pickle=False)
        d_pred = np.load(dr_pred_npz_path, allow_pickle=False)
        d_res = np.load(dr_res_npz_path, allow_pickle=False)

        # Load raw data
        X_raw = d_orig["X"].astype(np.float64, copy=False)           # (N, T, F)
        y_mask_raw = d_res["y_mask"].astype(np.int8, copy=False)      # (N, H)
        Y_pred_raw = d_pred["Y_pred"].astype(np.float64, copy=False)  # (N, H, D)
        Y_res_raw = d_res["Y_res"].astype(np.float64, copy=False)     # (N, H, D)
        pred_len_raw = d_orig.get("pred_len", y_mask_raw.sum(axis=1)).astype(np.int64)  # (N,)
        
        # Filter out sequences with extremely large residuals
        keep_mask = filter_outliers(Y_res_raw, outlier_threshold)
        n_original = len(keep_mask)
        n_kept = keep_mask.sum()
        n_filtered = n_original - n_kept
        
        if n_filtered > 0:
            print(f"Filtered out {n_filtered}/{n_original} sequences with residuals > {outlier_threshold}m")
            print(f"Keeping {n_kept} sequences ({n_kept/n_original*100:.1f}%)")
        
        # Apply filtering
        self.X = X_raw[keep_mask]
        self.y_mask = y_mask_raw[keep_mask]
        self.Y_pred = Y_pred_raw[keep_mask]
        self.Y_res = Y_res_raw[keep_mask]
        self.pred_len = pred_len_raw[keep_mask]

        if self.X.shape[0] != self.Y_pred.shape[0] or self.X.shape[0] != self.Y_res.shape[0]:
            raise ValueError("Sequence count mismatch across files")
        if self.Y_pred.shape[:2] != self.Y_res.shape[:2]:
            raise ValueError("Y_pred and Y_res must have same (N,H,*) shape")
        if self.y_mask.shape[0] != self.Y_pred.shape[0] or self.y_mask.shape[1] != self.Y_pred.shape[1]:
            raise ValueError("y_mask must match (N,H) of Y_pred/Y_res")

        self.norms = norms

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).to(torch.float32)            # (T, F)
        y_pred = torch.from_numpy(self.Y_pred[idx]).to(torch.float32)  # (H, D)
        y_res = torch.from_numpy(self.Y_res[idx]).to(torch.float32)    # (H, D)
        y_mask = torch.from_numpy(self.y_mask[idx]).to(torch.float32)  # (H,)
        pred_len = torch.tensor(self.pred_len[idx], dtype=torch.long)   # scalar

        if self.norms is not None:
            xm = torch.from_numpy(self.norms["x_mean"]).to(x.dtype); xs = torch.from_numpy(self.norms["x_std"]).to(x.dtype)
            pm = torch.from_numpy(self.norms["y_pred_mean"]).to(y_pred.dtype); ps = torch.from_numpy(self.norms["y_pred_std"]).to(y_pred.dtype)
            rm = torch.from_numpy(self.norms["y_res_mean"]).to(y_res.dtype);  rs = torch.from_numpy(self.norms["y_res_std"]).to(y_res.dtype)

            x = (x - xm) / (xs + 1e-6)
            y_pred = (y_pred - pm) / (ps + 1e-6)
            y_res = (y_res - rm) / (rs + 1e-6)

        return x, y_pred, y_res, y_mask, pred_len


def build_dataloaders(orig_dir: str, dr_out_dir: str, batch_size: int = 64, num_workers: int = 0,
                    shuffle_train: bool = True, normalize: bool = True,
                    scaler_pkl: str | None = None, outlier_threshold: float = 10000):
    def paths(split: str):
        return (os.path.join(orig_dir, f"{split}.npz"),
                os.path.join(dr_out_dir, f"{split}_Y_pred.npz"),
                os.path.join(dr_out_dir, f"{split}_Y_res.npz"))

    norms = None
    scalers = None
    if normalize:
        if scaler_pkl and os.path.exists(scaler_pkl):
            scalers = load_scalers(scaler_pkl)
        else:
            train_orig, train_pred, train_res = paths("train")
            scalers = fit_train_standard_scalers(train_orig, train_pred, train_res, outlier_threshold)
            if scaler_pkl:
                parent = os.path.dirname(scaler_pkl)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                save_scalers(scaler_pkl, scalers)

        norms = {
            "x_mean": scalers["x"].mean_.astype(np.float64),
            "x_std": scalers["x"].scale_.astype(np.float64),
            "y_pred_mean": scalers["y_pred"].mean_.astype(np.float64),
            "y_pred_std": scalers["y_pred"].scale_.astype(np.float64),
            "y_res_mean": scalers["y_res"].mean_.astype(np.float64),
            "y_res_std": scalers["y_res"].scale_.astype(np.float64),
        }

    loaders = {}
    for split in ["train", "val", "test"]:
        o, p, r = paths(split)
        ds = DRResidualDataset(o, p, r, norms=norms, outlier_threshold=outlier_threshold)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(shuffle_train and split == "train"),
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
    return loaders, scalers


def masked_mse(pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
    mask = y_mask.unsqueeze(-1)
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom


def compute_per_horizon_loss(pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor, 
                           loss_type: str = "mse", **loss_kwargs) -> torch.Tensor:
    """
    Compute loss for each prediction horizon step.
    Returns tensor of shape [H] with loss for each step.
    """
    B, H, D = pred.shape
    per_step_losses = []
    
    for h in range(H):
        pred_h = pred[:, h:h+1, :]  # [B, 1, D]
        target_h = target[:, h:h+1, :]  # [B, 1, D]
        mask_h = y_mask[:, h:h+1]  # [B, 1]
        
        if mask_h.sum() == 0:
            per_step_losses.append(torch.tensor(0.0, device=pred.device))
            continue
            
        if loss_type == "mse":
            loss_h = masked_mse(pred_h, target_h, mask_h)
        elif loss_type == "huber":
            loss_h = masked_huber(pred_h, target_h, mask_h, delta=loss_kwargs.get("delta", 1.0))
        elif loss_type == "tukey":
            loss_h = masked_tukey_biweight(pred_h, target_h, mask_h, c=loss_kwargs.get("c", 4.685))
        elif loss_type == "weighted_mse":
            loss_h = masked_weighted_mse(pred_h, target_h, mask_h, 
                                    outlier_threshold=loss_kwargs.get("outlier_threshold", 2.0),
                                    outlier_weight=loss_kwargs.get("outlier_weight", 0.1))
        elif loss_type == "weighted_huber":
            loss_h = masked_weighted_huber(pred_h, target_h, mask_h,
                                        delta=loss_kwargs.get("delta", 1.0),
                                        outlier_threshold=loss_kwargs.get("outlier_threshold", 2.0),
                                        outlier_weight=loss_kwargs.get("outlier_weight", 0.1))
        elif loss_type == "weighted_tukey":
            loss_h = masked_weighted_tukey_biweight(pred_h, target_h, mask_h,
                                                c=loss_kwargs.get("c", 4.685),
                                                outlier_threshold=loss_kwargs.get("outlier_threshold", 2.0),
                                                outlier_weight=loss_kwargs.get("outlier_weight", 0.1))
        else:
            loss_h = masked_mse(pred_h, target_h, mask_h)
            
        per_step_losses.append(loss_h)
    
    return torch.stack(per_step_losses)


def compute_per_predlen_loss(pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor, 
                           pred_lengths: torch.Tensor, loss_type: str = "mse", **loss_kwargs) -> dict:
    """
    Compute loss grouped by actual prediction lengths (5, 20, 60, 100, 140).
    Returns dict with prediction lengths as keys and average losses as values.
    """
    unique_pred_lens = pred_lengths.unique().cpu().numpy()
    pred_len_losses = {}
    
    for pred_len in unique_pred_lens:
        # Find sequences with this prediction length
        mask_indices = (pred_lengths == pred_len)
        if mask_indices.sum() == 0:
            continue
            
        # Extract sequences with this prediction length
        pred_subset = pred[mask_indices]  # [N_subset, H, D]
        target_subset = target[mask_indices]
        mask_subset = y_mask[mask_indices]
        
        # Only consider the first pred_len steps for these sequences
        pred_len = int(pred_len)
        pred_trunc = pred_subset[:, :pred_len, :]  # [N_subset, pred_len, D]
        target_trunc = target_subset[:, :pred_len, :]
        mask_trunc = mask_subset[:, :pred_len]
        
        # Compute loss for this prediction length group
        if loss_type == "mse":
            loss_val = masked_mse(pred_trunc, target_trunc, mask_trunc)
        elif loss_type == "huber":
            loss_val = masked_huber(pred_trunc, target_trunc, mask_trunc, delta=loss_kwargs.get("delta", 1.0))
        elif loss_type == "tukey":
            loss_val = masked_tukey_biweight(pred_trunc, target_trunc, mask_trunc, c=loss_kwargs.get("c", 4.685))
        elif loss_type == "weighted_mse":
            loss_val = masked_weighted_mse(pred_trunc, target_trunc, mask_trunc, 
                                    outlier_threshold=loss_kwargs.get("outlier_threshold", 2.0),
                                    outlier_weight=loss_kwargs.get("outlier_weight", 0.1))
        elif loss_type == "weighted_huber":
            loss_val = masked_weighted_huber(pred_trunc, target_trunc, mask_trunc,
                                        delta=loss_kwargs.get("delta", 1.0),
                                        outlier_threshold=loss_kwargs.get("outlier_threshold", 2.0),
                                        outlier_weight=loss_kwargs.get("outlier_weight", 0.1))
        elif loss_type == "weighted_tukey":
            loss_val = masked_weighted_tukey_biweight(pred_trunc, target_trunc, mask_trunc,
                                                c=loss_kwargs.get("c", 4.685),
                                                outlier_threshold=loss_kwargs.get("outlier_threshold", 2.0),
                                                outlier_weight=loss_kwargs.get("outlier_weight", 0.1))
        else:
            loss_val = masked_mse(pred_trunc, target_trunc, mask_trunc)
            
        pred_len_losses[pred_len] = loss_val.item()
    
    return pred_len_losses


def masked_huber(pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    mask = y_mask.unsqueeze(-1)
    r = pred - target
    abs_r = r.abs()
    quad = 0.5 * (r ** 2)
    lin = delta * (abs_r - 0.5 * delta)
    loss = torch.where(abs_r <= delta, quad, lin)
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_tukey_biweight(pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor, c: float = 4.685) -> torch.Tensor:
    mask = y_mask.unsqueeze(-1)
    r = pred - target
    u = (r / (c + 1e-12)).abs()
    inside = u <= 1.0
    # Tukey biweight loss per element: (c^2/6) * [1 - (1 - u^2)^3] for |u|<=1, else c^2/6
    one = torch.ones_like(u)
    val_inside = (c * c / 6.0) * (one - (one - u.pow(2)).pow(3))
    val_outside = (c * c) / 6.0
    loss = torch.where(inside, val_inside, torch.as_tensor(val_outside, dtype=val_inside.dtype, device=val_inside.device))
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def _compute_outlier_weights(pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor,
                        outlier_threshold: float, outlier_weight: float) -> torch.Tensor:
    """Helper to compute outlier weights based on residual magnitude."""
    mask = y_mask.unsqueeze(-1)
    r = pred - target
    r_masked = r * mask
    
    # Compute std of valid residuals for threshold
    mask_expanded = mask.expand_as(r_masked)
    valid_r = r_masked[mask_expanded.bool()]
    if valid_r.numel() > 0:
        r_std = valid_r.std() + 1e-6
        r_mean = valid_r.mean()
    else:
        r_std = 1.0
        r_mean = 0.0
    
    # Identify outliers based on standardized residual magnitude
    r_standardized = ((r - r_mean) / r_std).abs()
    is_outlier = r_standardized > outlier_threshold
    
    # Create weights: normal=1.0, outliers=outlier_weight
    weights = torch.where(is_outlier, torch.tensor(outlier_weight, device=r.device), torch.tensor(1.0, device=r.device))
    return weights * mask


def masked_weighted_mse(pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor, 
                    outlier_threshold: float = 2.0, outlier_weight: float = 0.1) -> torch.Tensor:
    """MSE with automatic outlier downweighting."""
    weights = _compute_outlier_weights(pred, target, y_mask, outlier_threshold, outlier_weight)
    r = pred - target
    loss = (r ** 2) * weights
    denom = weights.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_weighted_huber(pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor, delta: float = 1.0,
                        outlier_threshold: float = 2.0, outlier_weight: float = 0.1) -> torch.Tensor:
    """Huber loss with automatic outlier downweighting."""
    weights = _compute_outlier_weights(pred, target, y_mask, outlier_threshold, outlier_weight)
    r = pred - target
    abs_r = r.abs()
    quad = 0.5 * (r ** 2)
    lin = delta * (abs_r - 0.5 * delta)
    loss = torch.where(abs_r <= delta, quad, lin)
    loss = loss * weights
    denom = weights.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_weighted_tukey_biweight(pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor, c: float = 4.685,
                                outlier_threshold: float = 2.0, outlier_weight: float = 0.1) -> torch.Tensor:
    """Tukey biweight loss with automatic outlier downweighting."""
    weights = _compute_outlier_weights(pred, target, y_mask, outlier_threshold, outlier_weight)
    r = pred - target
    u = (r / (c + 1e-12)).abs()
    inside = u <= 1.0
    one = torch.ones_like(u)
    val_inside = (c * c / 6.0) * (one - (one - u.pow(2)).pow(3))
    val_outside = (c * c) / 6.0
    loss = torch.where(inside, val_inside, torch.as_tensor(val_outside, dtype=val_inside.dtype, device=val_inside.device))
    loss = loss * weights
    denom = weights.sum().clamp_min(1.0)
    return loss.sum() / denom


def check_alignment(orig_npz: str, ypred_npz: str, yres_npz: str, tol: float = 0.000001) -> bool:
    o = np.load(orig_npz, allow_pickle=False)
    p = np.load(ypred_npz, allow_pickle=False)
    r = np.load(yres_npz, allow_pickle=False)

    X = o["X"]
    Y_true = o["Y"]
    y_mask_o = o["y_mask"]

    Y_pred = p["Y_pred"]
    y_mask_p = p["y_mask"]

    Y_res = r["Y_res"]
    y_mask_r = r["y_mask"]

    assert X.shape[0] == Y_true.shape[0] == Y_pred.shape[0] == Y_res.shape[0]
    assert Y_true.shape == Y_pred.shape == Y_res.shape
    assert y_mask_o.shape == y_mask_p.shape == y_mask_r.shape == Y_true.shape[:2]
    assert np.array_equal(y_mask_o, y_mask_p)
    assert np.array_equal(y_mask_o, y_mask_r)

    m = y_mask_o.astype(bool)[..., None]
    err = np.abs(Y_true - (Y_pred + Y_res))
    max_err = float(err[m].max()) if m.any() else 0.0
    ok = max_err <= tol

    print(f"N={X.shape[0]}, T={X.shape[1]}, H={Y_true.shape[1]}, D={Y_true.shape[2]}")
    print(f"max|Y - (Y_pred+Y_res)| over mask = {max_err:.9f} (tol={tol})")
    print("OK" if ok else "FAIL")
    return ok


def check_alignment_splits(orig_dir: str, dr_out_dir: str, tol: float = 0.000001) -> dict:
    results = {}
    for split in ["train", "val", "test"]:
        results[split] = check_alignment(
            os.path.join(orig_dir, f"{split}.npz"),
            os.path.join(dr_out_dir, f"{split}_Y_pred.npz"),
            os.path.join(dr_out_dir, f"{split}_Y_res.npz"),
            tol,
        )
    return results


def fit_train_standard_scalers(orig_train_npz: str, dr_pred_train_npz: str, dr_res_train_npz: str, outlier_threshold: float = 10000) -> dict:
    o = np.load(orig_train_npz, allow_pickle=False)
    p = np.load(dr_pred_train_npz, allow_pickle=False)
    r = np.load(dr_res_train_npz, allow_pickle=False)

    X = o["X"].astype(np.float64)
    Y_pred = p["Y_pred"].astype(np.float64)
    Y_res  = r["Y_res"].astype(np.float64)
    y_mask = r["y_mask"].astype(np.int8)

    # Apply outlier filtering to training data
    keep_mask = filter_outliers(Y_res, outlier_threshold)
    if keep_mask.sum() < len(keep_mask):
        print(f"Filtering training data: keeping {keep_mask.sum()}/{len(keep_mask)} sequences")
        X = X[keep_mask]
        Y_pred = Y_pred[keep_mask]
        Y_res = Y_res[keep_mask]
        y_mask = y_mask[keep_mask]

    x_scaler = StandardScaler()
    x_scaler.fit(X.reshape(-1, X.shape[-1]))
    x_scaler.scale_[x_scaler.scale_ < 1e-6] = 1.0

    # Use 2D mask over (N,H) to select valid rows across the last dim
    mask2d = y_mask.astype(bool)
    if mask2d.any():
        yp_flat = Y_pred[mask2d]  # (K, D)
        yr_flat = Y_res[mask2d]   # (K, D)
    else:
        yp_flat = np.zeros((0, Y_pred.shape[-1]), dtype=np.float64)
        yr_flat = np.zeros((0, Y_res.shape[-1]), dtype=np.float64)

    y_pred_scaler = StandardScaler()
    y_res_scaler  = StandardScaler()
    if yp_flat.shape[0] > 0:
        y_pred_scaler.fit(yp_flat)
        y_pred_scaler.scale_[y_pred_scaler.scale_ < 1e-6] = 1.0
    else:
        y_pred_scaler.mean_ = np.zeros(Y_pred.shape[-1], dtype=np.float64)
        y_pred_scaler.scale_ = np.ones(Y_pred.shape[-1], dtype=np.float64)

    if yr_flat.shape[0] > 0:
        y_res_scaler.fit(yr_flat)
        y_res_scaler.scale_[y_res_scaler.scale_ < 1e-6] = 1.0
    else:
        y_res_scaler.mean_ = np.zeros(Y_res.shape[-1], dtype=np.float64)
        y_res_scaler.scale_ = np.ones(Y_res.shape[-1], dtype=np.float64)

    return {"x": x_scaler, "y_pred": y_pred_scaler, "y_res": y_res_scaler}


def save_scalers(pkl_path: str, scalers: dict):
    joblib.dump(scalers, pkl_path)


def load_scalers(pkl_path: str) -> dict:
    return joblib.load(pkl_path)


def unnormalize_y_pred_with_scaler(y_pred_norm: torch.Tensor, scalers: dict) -> torch.Tensor:
    s = scalers["y_pred"]
    mean = torch.from_numpy(s.mean_).to(y_pred_norm.dtype).to(y_pred_norm.device)
    scale = torch.from_numpy(s.scale_).to(y_pred_norm.dtype).to(y_pred_norm.device)
    return y_pred_norm * (scale + 1e-6) + mean


def unnormalize_y_res_with_scaler(y_res_norm: torch.Tensor, scalers: dict) -> torch.Tensor:
    s = scalers["y_res"]
    mean = torch.from_numpy(s.mean_).to(y_res_norm.dtype).to(y_res_norm.device)
    scale = torch.from_numpy(s.scale_).to(y_res_norm.dtype).to(y_res_norm.device)
    return y_res_norm * (scale + 1e-6) + mean