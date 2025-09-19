<<<<<<< HEAD
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def stats(name, arr):
    arr = np.asarray(arr)
    print(f"{name}: min={arr.min(0)}, max={arr.max(0)}, mean={arr.mean(0)}, std={arr.std(0)}")

# Pre-norm from CSVs
def csv_stats(csv_path, cols):
    df = pd.read_csv(csv_path)
    print(f"\n[Pre-norm CSV] {csv_path}")
    stats("  features", df[cols].values)

# Post-norm from sequences
def npz_stats(npzX_path):
    X = np.load(npzX_path)['X']
    print(f"\n[Post-norm NPZ] {npzX_path}")
    stats("  features", X.reshape(-1, X.shape[-1]))

if __name__ == "__main__":
    # edit paths if needed
    cols = ['lat','lon','baroaltitude','velocity','vertrate','delta_t','heading_sin','heading_cos']
    csv_stats("data/new/train.csv", cols)
    npz_stats("data/new_seq/trainX.npz")

    # scaler sanity
    try:
        with open("data/new/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        scaler = meta['scaler']
        print("\n[Scaler] type:", type(scaler).__name__)
        for attr in ('mean_', 'scale_', 'data_min_', 'data_max_'):
            if hasattr(scaler, attr):
                print(f"  {attr}:", getattr(scaler, attr))
    except Exception as e:
        print("[Scaler] no meta or failed:", e)
=======
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def stats(name, arr):
    arr = np.asarray(arr)
    print(f"{name}: min={arr.min(0)}, max={arr.max(0)}, mean={arr.mean(0)}, std={arr.std(0)}")

# Pre-norm from CSVs
def csv_stats(csv_path, cols):
    df = pd.read_csv(csv_path)
    print(f"\n[Pre-norm CSV] {csv_path}")
    stats("  features", df[cols].values)

# Post-norm from sequences
def npz_stats(npzX_path):
    X = np.load(npzX_path)['X']
    print(f"\n[Post-norm NPZ] {npzX_path}")
    stats("  features", X.reshape(-1, X.shape[-1]))

if __name__ == "__main__":
    # edit paths if needed
    cols = ['lat','lon','baroaltitude','velocity','vertrate','delta_t','heading_sin','heading_cos']
    csv_stats("data/new/train.csv", cols)
    npz_stats("data/new_seq/trainX.npz")

    # scaler sanity
    try:
        with open("data/new/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        scaler = meta['scaler']
        print("\n[Scaler] type:", type(scaler).__name__)
        for attr in ('mean_', 'scale_', 'data_min_', 'data_max_'):
            if hasattr(scaler, attr):
                print(f"  {attr}:", getattr(scaler, attr))
    except Exception as e:
        print("[Scaler] no meta or failed:", e)
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
