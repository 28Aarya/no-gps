"""
Dead Reckoning (DR) prediction using linear Euler integration.

Key changes:
- Uses linear integration: pos_next = pos_current + vel * dt
- Forward-fills NaN velocities from previous valid measurements
- Removed quadratic acceleration terms for better stability
- Simplified codebase by removing INS/acceleration models
"""
import numpy as np
import os
from typing import Dict, Tuple
from seq_dr import FEATURES as SEQ_FEATURES, TARGETS as SEQ_TARGETS

FEATURES = SEQ_FEATURES
TARGETS = SEQ_TARGETS

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, Y, y_mask from NPZ file with proper dtypes."""
    d = np.load(file_path, allow_pickle=True)
    X = d["X"].astype(np.float64, copy=False)
    Y = d["Y"].astype(np.float64, copy=False)
    y_mask = d["y_mask"].astype(np.int8, copy=False)
    return X, Y, y_mask

def map_feature_indices(feature_names: list) -> Dict[str, int]:
    """Create feature name to index mapping, validating all required features exist."""
    idx = {name: i for i, name in enumerate(feature_names)}
    for k in FEATURES:
        if k not in idx:
            raise KeyError(f"Missing feature '{k}' in feature_names: {feature_names}")
    return idx

def dead_reckon_step(prev_pos: np.ndarray, velocity_components: np.ndarray, dt: float) -> np.ndarray:
    """Compute next (x, y, z) position using linear Euler integration: pos_next = pos_current + vel * dt."""
    return prev_pos + velocity_components * dt

def dead_reckon_sequence(X: np.ndarray, pred_len: int, feat_map: dict, dt: float) -> np.ndarray:
    """Predict sequence using linear Euler integration with forward-filling for NaN velocities."""
    prev_pos = np.array([X[-1, feat_map["x"]], X[-1, feat_map["y"]], X[-1, feat_map["z"]]], dtype=np.float64)
    
    velocity_components = _get_valid_velocity(X, feat_map)
    
    out = np.zeros((pred_len, 3), dtype=np.float64)
    for t in range(pred_len):
        prev_pos = dead_reckon_step(prev_pos, velocity_components, dt)
        out[t] = prev_pos
    return out

def _get_valid_velocity(X: np.ndarray, feat_map: dict) -> np.ndarray:
    """Get valid velocity components with forward-filling for NaN/repeated positions."""
    v_x = X[-1, feat_map["v_x"]]
    v_y = X[-1, feat_map["v_y"]]
    v_z = X[-1, feat_map["v_z"]]
    
    if not np.isfinite(v_x) or not np.isfinite(v_y) or not np.isfinite(v_z):
        for i in range(X.shape[0] - 2, -1, -1):
            v_x_candidate = X[i, feat_map["v_x"]]
            v_y_candidate = X[i, feat_map["v_y"]]
            v_z_candidate = X[i, feat_map["v_z"]]
            
            if np.isfinite(v_x_candidate) and np.isfinite(v_y_candidate) and np.isfinite(v_z_candidate):
                v_x, v_y, v_z = v_x_candidate, v_y_candidate, v_z_candidate
                break
    
    return np.array([v_x, v_y, v_z], dtype=np.float64)

def run_dead_reckoning_split_fast(file_path: str, feat_map: dict, dt: float, out_dir: str, split_name: str, batch_size: int = 50000) -> Dict[str, str]:
    """Fast DR prediction with large batches."""
    print(f"Processing {split_name} with large batches...")
    
    with np.load(file_path) as data:
        total_sequences = len(data["X"])
        Hmax = data["Y"].shape[1]
        D = data["Y"].shape[2]
    
    print(f"Total sequences: {total_sequences}, Max horizon: {Hmax}, Features: {D}")
    
    all_y_pred = []
    all_y_res = []
    all_y_mask = []
    
    for batch_start in range(0, total_sequences, batch_size):
        batch_end = min(batch_start + batch_size, total_sequences)
        batch_size_actual = batch_end - batch_start
        
        print(f"  Processing batch {batch_start//batch_size + 1}/{(total_sequences + batch_size - 1)//batch_size} "
            f"(sequences {batch_start}-{batch_end-1})")
        
        with np.load(file_path) as data:
            X_batch = data["X"][batch_start:batch_end].astype(np.float64, copy=False)
            Y_true_batch = data["Y"][batch_start:batch_end].astype(np.float64, copy=False)
            y_mask_batch = data["y_mask"][batch_start:batch_end].astype(np.int8, copy=False)
        
        y_pred_batch = np.zeros_like(Y_true_batch, dtype=np.float64)
        
        for i in range(batch_size_actual):
            h_true = int(y_mask_batch[i].sum())
            if h_true > 0:
                dt_i = infer_dt_from_X(X_batch[i], feat_map, fallback=dt)
                seq_pred = dead_reckon_sequence(X_batch[i], h_true, feat_map, dt_i)
                y_pred_batch[i, :h_true, :] = seq_pred
        
        y_res_batch = Y_true_batch - y_pred_batch
        
        all_y_pred.append(y_pred_batch)
        all_y_res.append(y_res_batch)
        all_y_mask.append(y_mask_batch)
        
        del X_batch, Y_true_batch, y_mask_batch, y_pred_batch, y_res_batch
    
    print(f"  Saving results...")
    y_pred_final = np.concatenate(all_y_pred, axis=0)
    y_res_final = np.concatenate(all_y_res, axis=0)
    y_mask_final = np.concatenate(all_y_mask, axis=0)

    stats = rmse_by_horizon_from_res(y_res_final, y_mask_final)
    print("RMSE by horizon (steps):")
    for h, s in stats.items():
        print(f"  H={h}: rmse_all={s['rmse_all']:.6f}, x={s['rmse_x']:.6f}, y={s['rmse_y']:.6f}, z={s['rmse_z']:.6f}")

    y_pred_path = f"{out_dir}/{split_name}_Y_pred.npz"
    y_res_path = f"{out_dir}/{split_name}_Y_res.npz"

    np.savez_compressed(y_pred_path, Y_pred=y_pred_final, y_mask=y_mask_final)
    np.savez_compressed(y_res_path, Y_res=y_res_final, y_mask=y_mask_final)

    return {"y_pred": y_pred_path, "y_res": y_res_path}

def rmse_by_horizon_from_res(y_res: np.ndarray, y_mask: np.ndarray) -> dict:
    h_true = y_mask.sum(axis=1).astype(np.int64)
    stats = {}
    for h in np.unique(h_true):
        if h <= 0:
            continue
        idx = np.where(h_true == h)[0]
        if idx.size == 0:
            continue
        err2 = (y_res[idx, :h, :]) ** 2
        rmse_xyz = np.sqrt(err2.mean(axis=(0, 1)))
        rmse_all = float(np.sqrt(err2.mean()))
        stats[int(h)] = {
            "rmse_all": rmse_all,
            "rmse_x": float(rmse_xyz[0]),
            "rmse_y": float(rmse_xyz[1]),
            "rmse_z": float(rmse_xyz[2]),
        }
    return dict(sorted(stats.items()))

def validate_alignment_per_split(orig_npz_path: str, y_pred_path: str, y_res_path: str, split_name: str) -> bool:
    """Test that Y_pred and Y_res maintain exact alignment with original Y_true for one split."""
    orig = np.load(orig_npz_path)
    y_pred = np.load(y_pred_path)
    y_res = np.load(y_res_path)
    
    Y_true = orig["Y"]
    
    assert Y_true.shape == y_pred["Y_pred"].shape, f"{split_name} Y shape mismatch: orig={Y_true.shape}, y_pred={y_pred['Y_pred'].shape}"
    assert y_pred["Y_pred"].shape == y_res["Y_res"].shape, f"{split_name} Y_res shape mismatch: y_pred={y_pred['Y_pred'].shape}, y_res={y_res['Y_res'].shape}"
    
    assert np.array_equal(orig["y_mask"], y_pred["y_mask"]), f"{split_name} y_mask mismatch in y_pred"
    assert np.array_equal(y_pred["y_mask"], y_res["y_mask"]), f"{split_name} y_mask mismatch in y_res"
    
    assert y_pred["Y_pred"].dtype == np.float64, f"{split_name} Y_pred dtype: {y_pred['Y_pred'].dtype}"
    assert y_res["Y_res"].dtype == np.float64, f"{split_name} Y_res dtype: {y_res['Y_res'].dtype}"
    
    n_orig = len(Y_true)
    n_pred = len(y_pred["Y_pred"])
    n_res = len(y_res["Y_res"])
    assert n_orig == n_pred == n_res, f"{split_name} sequence count mismatch: orig={n_orig}, pred={n_pred}, res={n_res}"
    
    print(f"✓ {split_name} alignment validated:")
    print(f"  Y_true shape: {Y_true.shape}")
    print(f"  Y_pred shape: {y_pred['Y_pred'].shape}")
    print(f"  Y_res shape: {y_res['Y_res'].shape}")
    print(f"  Sequences: {n_orig}")
    return True

def run_dr_for_splits(orig_dir: str, out_dir: str, dt: float = 3.0, batch_size: int = 50000, run_sanity_checks: bool = True, sanity_sample_size: int = 1000) -> Dict[str, Dict[str, str]]:
    """Helper to run DR for all splits and generate Y_pred and Y_res files with large batch processing."""
    os.makedirs(out_dir, exist_ok=True)
    
    splits = ["train", "val", "test"]
    results = {}
    
    for split in splits:
        orig_path = f"{orig_dir}/{split}.npz"
        if os.path.exists(orig_path):
            print(f"\n Processing {split}...")
            
            feat_map = map_feature_indices(list(FEATURES))
            file_paths = run_dead_reckoning_split_fast(orig_path, feat_map, dt, out_dir, split, batch_size)
            
            print(f"Validating {split} alignment...")
            validate_alignment_per_split(orig_path, file_paths["y_pred"], file_paths["y_res"], split)
            
            results[split] = file_paths
            print(f"✓ Generated {split}: {file_paths['y_pred']}, {file_paths['y_res']}")
        else:
            print(f"⚠️  Skipping {split}: {orig_path} not found")
    
    return results

def infer_dt_from_X(X: np.ndarray, feat_map: dict, k: int = 5, max_dt: float = 10.0, fallback: float = 3.0) -> float:
    """Infer dt from last k timestamps in X, using median of positive deltas."""
    t_idx = feat_map["time"]
    times = X[-k:, t_idx].astype(np.float64)
    dts = np.diff(times)
    dts = dts[dts > 0]
    if dts.size == 0:
        return float(fallback)
    dt = float(np.median(dts))
    if not np.isfinite(dt) or dt <= 0 or dt > max_dt:
        return float(fallback)
    return dt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Dead Reckoning predictions")
    parser.add_argument("--orig_dir", required=True, help="Directory with train.npz, val.npz, test.npz")
    parser.add_argument("--out_dir", required=True, help="Output directory for DR results")
    parser.add_argument("--dt", type=float, default=3.0, help="Time step in seconds")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size for memory-efficient processing")
    parser.add_argument("--no_sanity_checks", action="store_true", help="Skip sanity checks")
    parser.add_argument("--sanity_sample_size", type=int, default=1000, help="Number of samples for sanity checks (default 1000)")
    
    args = parser.parse_args()
    run_dr_for_splits(args.orig_dir, args.out_dir, args.dt, args.batch_size, not args.no_sanity_checks, args.sanity_sample_size)