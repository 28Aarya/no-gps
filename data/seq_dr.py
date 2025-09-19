<<<<<<< HEAD
"""Sequence generator for Dead Reckoning (DR) training and evaluation.

Generates (X, Y) pairs strictly within group boundaries, choosing the largest
prediction horizon that fits at each start. Short flights are kept
and get shorter horizons automatically.

Inputs DataFrame columns (required):
- One grouping key: prefer 'group_id', fallback to 'flight_id'
- Feature columns in fixed order: time, gap_flag, heading_sin, heading_cos, v_x, v_y, v_z, x, y, z
- 'time' column (UNIX seconds) is used for sorting within groups

Outputs:
- List of (X, Y) pairs where X has shape (past_window, num_features) and Y has
shape (H, num_targets) for the assigned prediction length H. All arrays are
float64.
"""

from typing import List, Sequence, Tuple, Iterator
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


FEATURES: Tuple[str, ...] = (
    "time",
    "gap_flag", 
    "heading_sin",
    "heading_cos",
    "v_x",
    "v_y", 
    "v_z",
    "x",
    "y",
    "z",
)

# Targets for Y (prediction) – x, y, z coordinates
TARGETS: Tuple[str, ...] = (
    "x",
    "y",
    "z",
)


def _get_group_column(df):
    """Get the appropriate group column name from DataFrame.
    
    Returns 'group_id' if present, otherwise 'flight_id'.
    """
    return "group_id" if "group_id" in df.columns else "flight_id"


def _validate_columns(df):
    group_col = _get_group_column(df)
    required = {group_col, *FEATURES}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def generate_sequences(
    df,
    past_window,
    pred_len=(5, 20, 60, 100, 140),
):
    """Generate (X, Y) sequences within each group_id, choosing the largest horizon
    that fits at each start. Short flights are kept (they just get shorter horizons).
    
    OPTIMIZED VERSION with major performance improvements:
    1. Pre-sort DataFrame once instead of per-group
    2. Convert to numpy arrays once outside loops
    3. Use direct indexing instead of DataFrame.loc
    4. Progress tracking for large datasets
    """
    _validate_columns(df)

    pred_len = tuple(sorted(int(h) for h in pred_len))
    min_h = pred_len[0]
    group_col = _get_group_column(df)

    print("[OPTIMIZED] Processing {} rows with {} groups...".format(len(df), df[group_col].nunique()))
    start_time = time.time()

    if "time" in df.columns:
        print("[OPTIMIZED] Pre-sorting by {} and time...".format(group_col))
        df = df.sort_values([group_col, "time"]).reset_index(drop=True)
    
    # OPTIMIZATION 2: Convert to numpy arrays once, outside the loop
    print(f"[OPTIMIZED] Converting to numpy arrays...")
    x_mat = df[list(FEATURES)].to_numpy(dtype="float64", copy=False)
    y_mat = df[list(TARGETS)].to_numpy(dtype="float64", copy=False)
    
    sequences: List[Tuple[np.ndarray, np.ndarray]] = []
    
    # Count total groups for progress bar
    total_groups = df[group_col].nunique()
    
    # OPTIMIZATION 3: Use groupby with pre-sorted data and progress tracking
    group_iterator = df.groupby(group_col, sort=False)
    if total_groups > 100:  # Only show progress for large datasets
        group_iterator = tqdm(group_iterator, total=total_groups, desc="Processing groups")
    
    for group_name, group in group_iterator:
        # Get start and end indices for this group
        start_idx = group.index[0]
        end_idx = group.index[-1] + 1
        
        # Extract group data using indices (much faster than loc)
        x_mat_g = x_mat[start_idx:end_idx]
        y_mat_g = y_mat[start_idx:end_idx]
        
        n = len(group)
        max_start = n - (past_window + min_h)
        if max_start < 0:
            continue

        # Generate sequences for this group with NO overlap
        start = 0
        while start <= max_start:
            remaining = n - (start + past_window)
            # pick largest horizon that fits
            h = 0
            for H in reversed(pred_len):
                if H <= remaining:
                    h = H
                    break
            if h <= 0:
                break

            # Direct numpy slicing (no DataFrame access)
            X = x_mat_g[start : start + past_window, :]
            Y = y_mat_g[start + past_window : start + past_window + h, :]
            sequences.append((X, Y))

            # advance start to avoid any overlap between consecutive sequences
            start += past_window + h

    elapsed = time.time() - start_time
    print("[OPTIMIZED] Generated {} sequences in {:.2f} seconds".format(len(sequences), elapsed))
    if elapsed > 0:
        print("[OPTIMIZED] Average: {:.1f} sequences/second".format(len(sequences)/elapsed))
    
    return sequences


__all__ = ["generate_sequences", "generate_sequences_iter", "FEATURES", "TARGETS", "save_sequences_npz_single", "save_sequences_npz_sharded", "generate_save_sequences", "summarize_npz", "summarize_saved_sequences"]


# ------------------------------ NPZ (single-file) ----------------------------


def save_sequences_npz_single(
    sequences,
    out_dir,
    split_name,
    max_pred_len=None,
):
    os.makedirs(out_dir, exist_ok=True)
    if not sequences:
        path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez_compressed(path, X=np.empty((0,0,0)), Y=np.empty((0,0,0)), y_mask=np.empty((0,0), dtype=np.int8), pred_len=np.empty((0,), dtype=np.int64))
        return path

    print("[OPTIMIZED] Saving {} sequences...".format(len(sequences)))
    start_time = time.time()

    # OPTIMIZATION: Pre-allocate arrays for better performance
    N = len(sequences)
    pred_len_arr = np.array([y.shape[0] for _, y in sequences], dtype=np.int64)
    Hmax = int(max_pred_len) if max_pred_len is not None else int(pred_len_arr.max())
    
    # Get dimensions from first sequence
    Fx = sequences[0][0].shape[-1]  # X features
    Fy = sequences[0][1].shape[-1]  # Y features (should be 3 for x, y, z)
    
    print("[OPTIMIZED] X shape: (N={}, past_window={}, features={})".format(N, sequences[0][0].shape[0], Fx))
    print("[OPTIMIZED] Y shape: (N={}, max_horizon={}, targets={})".format(N, Hmax, Fy))
    
    # Pre-allocate output arrays
    X_all = np.zeros((N, sequences[0][0].shape[0], Fx), dtype="float64")
    Y_pad = np.zeros((N, Hmax, Fy), dtype="float64")
    y_mask = np.zeros((N, Hmax), dtype=np.int8)
    
    # OPTIMIZATION: Fill arrays in a single loop (no list comprehensions)
    for i, (X, Y) in enumerate(sequences):
        X_all[i] = X
        h = min(Y.shape[0], Hmax)
        if h > 0:
            Y_pad[i, :h, :] = Y
            y_mask[i, :h] = 1

    # Additional assertion to ensure final arrays have same number of sequences
    assert X_all.shape[0] == Y_pad.shape[0] == y_mask.shape[0], f"Final array sequence counts must match: X={X_all.shape[0]}, Y={Y_pad.shape[0]}, y_mask={y_mask.shape[0]}"

    path = os.path.join(out_dir, f"{split_name}.npz")
    np.savez_compressed(path, X=X_all, Y=Y_pad, y_mask=y_mask, pred_len=pred_len_arr)
    
    elapsed = time.time() - start_time
    print("[OPTIMIZED] Saved to {} in {:.2f} seconds".format(path, elapsed))
    
    return path


def generate_sequences_iter(
    df,
    past_window,
    pred_len=(5, 20, 60, 100, 140),
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    _validate_columns(df)

    pred_len = tuple(sorted(int(h) for h in pred_len))
    min_h = pred_len[0]
    group_col = _get_group_column(df)

    if "time" in df.columns:
        df = df.sort_values([group_col, "time"]).reset_index(drop=True)

    x_mat = df[list(FEATURES)].to_numpy(dtype="float64", copy=False)
    y_mat = df[list(TARGETS)].to_numpy(dtype="float64", copy=False)

    group_iterator = df.groupby(group_col, sort=False)

    for _, group in group_iterator:
        start_idx = group.index[0]
        end_idx = group.index[-1] + 1

        x_mat_g = x_mat[start_idx:end_idx]
        y_mat_g = y_mat[start_idx:end_idx]

        n = len(group)
        max_start = n - (past_window + min_h)
        if max_start < 0:
            continue

        start = 0
        while start <= max_start:
            remaining = n - (start + past_window)
            h = 0
            for H in reversed(pred_len):
                if H <= remaining:
                    h = H
                    break
            if h <= 0:
                break

            X = x_mat_g[start : start + past_window, :]
            Y = y_mat_g[start + past_window : start + past_window + h, :]
            yield (X, Y)

            start += past_window + h


def save_sequences_npz_sharded(
    sequences_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
    out_dir: str,
    split_name: str,
    max_pred_len: int,
    max_sequences_per_shard: int = 10000,
):
    os.makedirs(out_dir, exist_ok=True)

    shard_idx = 0
    saved_paths: List[str] = []
    while True:
        batch_X: List[np.ndarray] = []
        batch_Y: List[np.ndarray] = []
        try:
            for _ in range(max_sequences_per_shard):
                X, Y = next(sequences_iter)
                batch_X.append(X)
                batch_Y.append(Y)
        except StopIteration:
            pass

        if not batch_X:
            break

        N = len(batch_X)
        Fx = batch_X[0].shape[-1]
        past_window_len = batch_X[0].shape[0]
        Fy = batch_Y[0].shape[-1]
        Hmax = int(max_pred_len)

        X_all = np.zeros((N, past_window_len, Fx), dtype="float64")
        Y_pad = np.zeros((N, Hmax, Fy), dtype="float64")
        y_mask = np.zeros((N, Hmax), dtype=np.int8)
        pred_len_arr = np.zeros((N,), dtype=np.int64)

        for i, (X, Y) in enumerate(zip(batch_X, batch_Y)):
            X_all[i] = X
            h = min(Y.shape[0], Hmax)
            pred_len_arr[i] = h
            if h > 0:
                Y_pad[i, :h, :] = Y
                y_mask[i, :h] = 1

        path = os.path.join(out_dir, f"{split_name}.part{shard_idx:03d}.npz")
        np.savez_compressed(path, X=X_all, Y=Y_pad, y_mask=y_mask, pred_len=pred_len_arr)
        saved_paths.append(path)
        shard_idx += 1

    if not saved_paths:
        # Save an empty shard for consistency
        path = os.path.join(out_dir, f"{split_name}.part000.npz")
        np.savez_compressed(
            path,
            X=np.empty((0,0,0)),
            Y=np.empty((0,0,0)),
            y_mask=np.empty((0,0), dtype=np.int8),
            pred_len=np.empty((0,), dtype=np.int64),
        )
        saved_paths.append(path)

    return saved_paths

def generate_save_sequences(
    train_csv,
    out_dir,
    past_window,
    pred_len=(5, 20, 60, 100, 140),
    val_csv=None,
    test_csv=None,
    max_sequences_per_shard=10000,
):
    results = {}

    def _one(csv_path, name):
        print("\n[OPTIMIZED] Processing {} split: {}".format(name, csv_path))
        df = pd.read_csv(csv_path, low_memory=False)

        group_col = _get_group_column(df)
        if group_col == "flight_id" and "group_id" not in df.columns:
            df["group_id"] = df["flight_id"]
            group_col = "group_id"

        print("[{}] Feature order for X: {}".format(name, FEATURES))
        print("[{}] Target order for Y: {}".format(name, TARGETS))

        if max_sequences_per_shard and max_sequences_per_shard > 0:
            seq_iter = generate_sequences_iter(df, past_window=past_window, pred_len=pred_len)
            paths = save_sequences_npz_sharded(
                sequences_iter=iter(seq_iter),
                out_dir=out_dir,
                split_name=name,
                max_pred_len=max(pred_len),
                max_sequences_per_shard=int(max_sequences_per_shard),
            )
            return paths
        else:
            seqs = generate_sequences(df, past_window=past_window, pred_len=pred_len)
            return save_sequences_npz_single(
                seqs,
                out_dir=out_dir,
                split_name=name,
                max_pred_len=max(pred_len),
            )

    results["train"] = _one(train_csv, "train")
    if val_csv:
        results["val"] = _one(val_csv, "val")
    if test_csv:
        results["test"] = _one(test_csv, "test")
    return results


# ------------------------------- Inspect helpers -----------------------------

def summarize_npz(npz_path, show_samples=1):
    """Print shapes and a few mapped samples from a saved NPZ.

    Shows one or more (X last step, Y valid steps, y_mask) mapped to FEATURES.
    """
    d = np.load(npz_path, allow_pickle=True)
    X, Y, y_mask = d["X"], d["Y"], d["y_mask"]
    print("{}: X{}, Y{}, y_mask{}".format(npz_path, X.shape, Y.shape, y_mask.shape))
    pred_lens = y_mask.sum(axis=1)
    for i in range(min(show_samples, X.shape[0])):
        h = int(pred_lens[i])
        print("sample {} pred_len={}".format(i, h))
        print("FEATURES:", FEATURES)
        print("X[-1] mapped:", dict(zip(FEATURES, X[i, -1].tolist())))
        print("Y[:h] mapped:", [dict(zip(TARGETS, row.tolist())) for row in Y[i, :h]])
        print("y_mask:", y_mask[i].tolist())


def summarize_saved_sequences(out_dir, splits=("train", "val", "test")):
    """Summarize train/val/test NPZs in out_dir.

    Prints shapes and a sample from each split if present.
    """
    import glob as _glob
    for name in splits:
        shard_paths = sorted(_glob.glob(os.path.join(out_dir, f"{name}.part*.npz")))
        single_path = os.path.join(out_dir, f"{name}.npz")
        if shard_paths:
            print(f"Found {len(shard_paths)} shards for split '{name}'. Showing first shard:")
            summarize_npz(shard_paths[0], show_samples=1)
        elif os.path.exists(single_path):
            summarize_npz(single_path, show_samples=1)
        else:
            print("Missing: {} or shards".format(single_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OPTIMIZED: Generate padded DR sequences (X, Y, y_mask) and save NPZs per split.")
    parser.add_argument("--train_csv", required=True, help="Path to train.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for train/val/test NPZs")
    parser.add_argument("--val_csv", help="Path to val.csv")
    parser.add_argument("--test_csv", help="Path to test.csv")
    parser.add_argument("--past_window", type=int, default=60, help="Past window length for X (default 60)")
    parser.add_argument("--pred_len", default="5,20,60,100,140", help="Comma-separated prediction lengths, e.g. 5,20,60,100,140")
    parser.add_argument("--summarize", action="store_true", help="Print shapes and one sample per split after saving")
    parser.add_argument("--max_sequences_per_shard", type=int, default=10000, help="Max sequences per shard NPZ (default 10000). Set 0 to save single file.")
    args = parser.parse_args()

    pred_len_list = tuple(int(x) for x in str(args.pred_len).split(",") if x.strip())

    print("[OPTIMIZED] Starting sequence generation...")
    print("[OPTIMIZED] Past window: {}".format(args.past_window))
    print("[OPTIMIZED] Prediction lengths: {}".format(pred_len_list))

    generate_save_sequences(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        past_window=args.past_window,
        pred_len=pred_len_list,
        max_sequences_per_shard=args.max_sequences_per_shard,
    )

    if args.summarize:
        summarize_saved_sequences(args.out_dir)

    print("[OPTIMIZED] Sequence generation completed!")
=======
"""Sequence generator for Dead Reckoning (DR) training and evaluation.

Generates (X, Y) pairs strictly within group boundaries, choosing the largest
prediction horizon that fits at each start. Short flights are kept
and get shorter horizons automatically.

Inputs DataFrame columns (required):
- One grouping key: prefer 'group_id', fallback to 'flight_id'
- Feature columns in fixed order: time, gap_flag, heading_sin, heading_cos, v_x, v_y, v_z, x, y, z
- 'time' column (UNIX seconds) is used for sorting within groups

Outputs:
- List of (X, Y) pairs where X has shape (past_window, num_features) and Y has
shape (H, num_targets) for the assigned prediction length H. All arrays are
float64.
"""

from typing import List, Sequence, Tuple, Iterator
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


FEATURES: Tuple[str, ...] = (
    "time",
    "gap_flag", 
    "heading_sin",
    "heading_cos",
    "v_x",
    "v_y", 
    "v_z",
    "x",
    "y",
    "z",
)

# Targets for Y (prediction) – x, y, z coordinates
TARGETS: Tuple[str, ...] = (
    "x",
    "y",
    "z",
)


def _get_group_column(df):
    """Get the appropriate group column name from DataFrame.
    
    Returns 'group_id' if present, otherwise 'flight_id'.
    """
    return "group_id" if "group_id" in df.columns else "flight_id"


def _validate_columns(df):
    group_col = _get_group_column(df)
    required = {group_col, *FEATURES}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def generate_sequences(
    df,
    past_window,
    pred_len=(5, 20, 60, 100, 140),
):
    """Generate (X, Y) sequences within each group_id, choosing the largest horizon
    that fits at each start. Short flights are kept (they just get shorter horizons).
    
    OPTIMIZED VERSION with major performance improvements:
    1. Pre-sort DataFrame once instead of per-group
    2. Convert to numpy arrays once outside loops
    3. Use direct indexing instead of DataFrame.loc
    4. Progress tracking for large datasets
    """
    _validate_columns(df)

    pred_len = tuple(sorted(int(h) for h in pred_len))
    min_h = pred_len[0]
    group_col = _get_group_column(df)

    print("[OPTIMIZED] Processing {} rows with {} groups...".format(len(df), df[group_col].nunique()))
    start_time = time.time()

    if "time" in df.columns:
        print("[OPTIMIZED] Pre-sorting by {} and time...".format(group_col))
        df = df.sort_values([group_col, "time"]).reset_index(drop=True)
    
    # OPTIMIZATION 2: Convert to numpy arrays once, outside the loop
    print(f"[OPTIMIZED] Converting to numpy arrays...")
    x_mat = df[list(FEATURES)].to_numpy(dtype="float64", copy=False)
    y_mat = df[list(TARGETS)].to_numpy(dtype="float64", copy=False)
    
    sequences: List[Tuple[np.ndarray, np.ndarray]] = []
    
    # Count total groups for progress bar
    total_groups = df[group_col].nunique()
    
    # OPTIMIZATION 3: Use groupby with pre-sorted data and progress tracking
    group_iterator = df.groupby(group_col, sort=False)
    if total_groups > 100:  # Only show progress for large datasets
        group_iterator = tqdm(group_iterator, total=total_groups, desc="Processing groups")
    
    for group_name, group in group_iterator:
        # Get start and end indices for this group
        start_idx = group.index[0]
        end_idx = group.index[-1] + 1
        
        # Extract group data using indices (much faster than loc)
        x_mat_g = x_mat[start_idx:end_idx]
        y_mat_g = y_mat[start_idx:end_idx]
        
        n = len(group)
        max_start = n - (past_window + min_h)
        if max_start < 0:
            continue

        # Generate sequences for this group with NO overlap
        start = 0
        while start <= max_start:
            remaining = n - (start + past_window)
            # pick largest horizon that fits
            h = 0
            for H in reversed(pred_len):
                if H <= remaining:
                    h = H
                    break
            if h <= 0:
                break

            # Direct numpy slicing (no DataFrame access)
            X = x_mat_g[start : start + past_window, :]
            Y = y_mat_g[start + past_window : start + past_window + h, :]
            sequences.append((X, Y))

            # advance start to avoid any overlap between consecutive sequences
            start += past_window + h

    elapsed = time.time() - start_time
    print("[OPTIMIZED] Generated {} sequences in {:.2f} seconds".format(len(sequences), elapsed))
    if elapsed > 0:
        print("[OPTIMIZED] Average: {:.1f} sequences/second".format(len(sequences)/elapsed))
    
    return sequences


__all__ = ["generate_sequences", "generate_sequences_iter", "FEATURES", "TARGETS", "save_sequences_npz_single", "save_sequences_npz_sharded", "generate_save_sequences", "summarize_npz", "summarize_saved_sequences"]


# ------------------------------ NPZ (single-file) ----------------------------


def save_sequences_npz_single(
    sequences,
    out_dir,
    split_name,
    max_pred_len=None,
):
    os.makedirs(out_dir, exist_ok=True)
    if not sequences:
        path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez_compressed(path, X=np.empty((0,0,0)), Y=np.empty((0,0,0)), y_mask=np.empty((0,0), dtype=np.int8), pred_len=np.empty((0,), dtype=np.int64))
        return path

    print("[OPTIMIZED] Saving {} sequences...".format(len(sequences)))
    start_time = time.time()

    # OPTIMIZATION: Pre-allocate arrays for better performance
    N = len(sequences)
    pred_len_arr = np.array([y.shape[0] for _, y in sequences], dtype=np.int64)
    Hmax = int(max_pred_len) if max_pred_len is not None else int(pred_len_arr.max())
    
    # Get dimensions from first sequence
    Fx = sequences[0][0].shape[-1]  # X features
    Fy = sequences[0][1].shape[-1]  # Y features (should be 3 for x, y, z)
    
    print("[OPTIMIZED] X shape: (N={}, past_window={}, features={})".format(N, sequences[0][0].shape[0], Fx))
    print("[OPTIMIZED] Y shape: (N={}, max_horizon={}, targets={})".format(N, Hmax, Fy))
    
    # Pre-allocate output arrays
    X_all = np.zeros((N, sequences[0][0].shape[0], Fx), dtype="float64")
    Y_pad = np.zeros((N, Hmax, Fy), dtype="float64")
    y_mask = np.zeros((N, Hmax), dtype=np.int8)
    
    # OPTIMIZATION: Fill arrays in a single loop (no list comprehensions)
    for i, (X, Y) in enumerate(sequences):
        X_all[i] = X
        h = min(Y.shape[0], Hmax)
        if h > 0:
            Y_pad[i, :h, :] = Y
            y_mask[i, :h] = 1

    # Additional assertion to ensure final arrays have same number of sequences
    assert X_all.shape[0] == Y_pad.shape[0] == y_mask.shape[0], f"Final array sequence counts must match: X={X_all.shape[0]}, Y={Y_pad.shape[0]}, y_mask={y_mask.shape[0]}"

    path = os.path.join(out_dir, f"{split_name}.npz")
    np.savez_compressed(path, X=X_all, Y=Y_pad, y_mask=y_mask, pred_len=pred_len_arr)
    
    elapsed = time.time() - start_time
    print("[OPTIMIZED] Saved to {} in {:.2f} seconds".format(path, elapsed))
    
    return path


def generate_sequences_iter(
    df,
    past_window,
    pred_len=(5, 20, 60, 100, 140),
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    _validate_columns(df)

    pred_len = tuple(sorted(int(h) for h in pred_len))
    min_h = pred_len[0]
    group_col = _get_group_column(df)

    if "time" in df.columns:
        df = df.sort_values([group_col, "time"]).reset_index(drop=True)

    x_mat = df[list(FEATURES)].to_numpy(dtype="float64", copy=False)
    y_mat = df[list(TARGETS)].to_numpy(dtype="float64", copy=False)

    group_iterator = df.groupby(group_col, sort=False)

    for _, group in group_iterator:
        start_idx = group.index[0]
        end_idx = group.index[-1] + 1

        x_mat_g = x_mat[start_idx:end_idx]
        y_mat_g = y_mat[start_idx:end_idx]

        n = len(group)
        max_start = n - (past_window + min_h)
        if max_start < 0:
            continue

        start = 0
        while start <= max_start:
            remaining = n - (start + past_window)
            h = 0
            for H in reversed(pred_len):
                if H <= remaining:
                    h = H
                    break
            if h <= 0:
                break

            X = x_mat_g[start : start + past_window, :]
            Y = y_mat_g[start + past_window : start + past_window + h, :]
            yield (X, Y)

            start += past_window + h


def save_sequences_npz_sharded(
    sequences_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
    out_dir: str,
    split_name: str,
    max_pred_len: int,
    max_sequences_per_shard: int = 10000,
):
    os.makedirs(out_dir, exist_ok=True)

    shard_idx = 0
    saved_paths: List[str] = []
    while True:
        batch_X: List[np.ndarray] = []
        batch_Y: List[np.ndarray] = []
        try:
            for _ in range(max_sequences_per_shard):
                X, Y = next(sequences_iter)
                batch_X.append(X)
                batch_Y.append(Y)
        except StopIteration:
            pass

        if not batch_X:
            break

        N = len(batch_X)
        Fx = batch_X[0].shape[-1]
        past_window_len = batch_X[0].shape[0]
        Fy = batch_Y[0].shape[-1]
        Hmax = int(max_pred_len)

        X_all = np.zeros((N, past_window_len, Fx), dtype="float64")
        Y_pad = np.zeros((N, Hmax, Fy), dtype="float64")
        y_mask = np.zeros((N, Hmax), dtype=np.int8)
        pred_len_arr = np.zeros((N,), dtype=np.int64)

        for i, (X, Y) in enumerate(zip(batch_X, batch_Y)):
            X_all[i] = X
            h = min(Y.shape[0], Hmax)
            pred_len_arr[i] = h
            if h > 0:
                Y_pad[i, :h, :] = Y
                y_mask[i, :h] = 1

        path = os.path.join(out_dir, f"{split_name}.part{shard_idx:03d}.npz")
        np.savez_compressed(path, X=X_all, Y=Y_pad, y_mask=y_mask, pred_len=pred_len_arr)
        saved_paths.append(path)
        shard_idx += 1

    if not saved_paths:
        # Save an empty shard for consistency
        path = os.path.join(out_dir, f"{split_name}.part000.npz")
        np.savez_compressed(
            path,
            X=np.empty((0,0,0)),
            Y=np.empty((0,0,0)),
            y_mask=np.empty((0,0), dtype=np.int8),
            pred_len=np.empty((0,), dtype=np.int64),
        )
        saved_paths.append(path)

    return saved_paths

def generate_save_sequences(
    train_csv,
    out_dir,
    past_window,
    pred_len=(5, 20, 60, 100, 140),
    val_csv=None,
    test_csv=None,
    max_sequences_per_shard=10000,
):
    results = {}

    def _one(csv_path, name):
        print("\n[OPTIMIZED] Processing {} split: {}".format(name, csv_path))
        df = pd.read_csv(csv_path, low_memory=False)

        group_col = _get_group_column(df)
        if group_col == "flight_id" and "group_id" not in df.columns:
            df["group_id"] = df["flight_id"]
            group_col = "group_id"

        print("[{}] Feature order for X: {}".format(name, FEATURES))
        print("[{}] Target order for Y: {}".format(name, TARGETS))

        if max_sequences_per_shard and max_sequences_per_shard > 0:
            seq_iter = generate_sequences_iter(df, past_window=past_window, pred_len=pred_len)
            paths = save_sequences_npz_sharded(
                sequences_iter=iter(seq_iter),
                out_dir=out_dir,
                split_name=name,
                max_pred_len=max(pred_len),
                max_sequences_per_shard=int(max_sequences_per_shard),
            )
            return paths
        else:
            seqs = generate_sequences(df, past_window=past_window, pred_len=pred_len)
            return save_sequences_npz_single(
                seqs,
                out_dir=out_dir,
                split_name=name,
                max_pred_len=max(pred_len),
            )

    results["train"] = _one(train_csv, "train")
    if val_csv:
        results["val"] = _one(val_csv, "val")
    if test_csv:
        results["test"] = _one(test_csv, "test")
    return results


# ------------------------------- Inspect helpers -----------------------------

def summarize_npz(npz_path, show_samples=1):
    """Print shapes and a few mapped samples from a saved NPZ.

    Shows one or more (X last step, Y valid steps, y_mask) mapped to FEATURES.
    """
    d = np.load(npz_path, allow_pickle=True)
    X, Y, y_mask = d["X"], d["Y"], d["y_mask"]
    print("{}: X{}, Y{}, y_mask{}".format(npz_path, X.shape, Y.shape, y_mask.shape))
    pred_lens = y_mask.sum(axis=1)
    for i in range(min(show_samples, X.shape[0])):
        h = int(pred_lens[i])
        print("sample {} pred_len={}".format(i, h))
        print("FEATURES:", FEATURES)
        print("X[-1] mapped:", dict(zip(FEATURES, X[i, -1].tolist())))
        print("Y[:h] mapped:", [dict(zip(TARGETS, row.tolist())) for row in Y[i, :h]])
        print("y_mask:", y_mask[i].tolist())


def summarize_saved_sequences(out_dir, splits=("train", "val", "test")):
    """Summarize train/val/test NPZs in out_dir.

    Prints shapes and a sample from each split if present.
    """
    import glob as _glob
    for name in splits:
        shard_paths = sorted(_glob.glob(os.path.join(out_dir, f"{name}.part*.npz")))
        single_path = os.path.join(out_dir, f"{name}.npz")
        if shard_paths:
            print(f"Found {len(shard_paths)} shards for split '{name}'. Showing first shard:")
            summarize_npz(shard_paths[0], show_samples=1)
        elif os.path.exists(single_path):
            summarize_npz(single_path, show_samples=1)
        else:
            print("Missing: {} or shards".format(single_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OPTIMIZED: Generate padded DR sequences (X, Y, y_mask) and save NPZs per split.")
    parser.add_argument("--train_csv", required=True, help="Path to train.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for train/val/test NPZs")
    parser.add_argument("--val_csv", help="Path to val.csv")
    parser.add_argument("--test_csv", help="Path to test.csv")
    parser.add_argument("--past_window", type=int, default=60, help="Past window length for X (default 60)")
    parser.add_argument("--pred_len", default="5,20,60,100,140", help="Comma-separated prediction lengths, e.g. 5,20,60,100,140")
    parser.add_argument("--summarize", action="store_true", help="Print shapes and one sample per split after saving")
    parser.add_argument("--max_sequences_per_shard", type=int, default=10000, help="Max sequences per shard NPZ (default 10000). Set 0 to save single file.")
    args = parser.parse_args()

    pred_len_list = tuple(int(x) for x in str(args.pred_len).split(",") if x.strip())

    print("[OPTIMIZED] Starting sequence generation...")
    print("[OPTIMIZED] Past window: {}".format(args.past_window))
    print("[OPTIMIZED] Prediction lengths: {}".format(pred_len_list))

    generate_save_sequences(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        past_window=args.past_window,
        pred_len=pred_len_list,
        max_sequences_per_shard=args.max_sequences_per_shard,
    )

    if args.summarize:
        summarize_saved_sequences(args.out_dir)

    print("[OPTIMIZED] Sequence generation completed!")
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
