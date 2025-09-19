<<<<<<< HEAD
import numpy as np, os

def check(split, orig_dir="seq_ecef", dr_dir="dr_results"):
    o = np.load(os.path.join(orig_dir, f"{split}.npz"))
    p = np.load(os.path.join(dr_dir, f"{split}_Y_pred.npz"))
    r = np.load(os.path.join(dr_dir, f"{split}_Y_res.npz"))

    X, Y, y_mask_o = o["X"], o["Y"], o["y_mask"]
    Yp, y_mask_p   = p["Y_pred"], p["y_mask"]
    Yr, y_mask_r   = r["Y_res"],  r["y_mask"]

    print(f"\n[{split}]")
    print("N (X,Y):", X.shape[0], Y.shape[0])
    print("N (Y_pred):", Yp.shape[0], "N (Y_res):", Yr.shape[0])
    print("Y shapes:", Y.shape, Yp.shape, Yr.shape)
    print("mask shapes:", y_mask_o.shape, y_mask_p.shape, y_mask_r.shape)

    assert X.shape[0] == Y.shape[0] == Yp.shape[0] == Yr.shape[0], "Sequence count mismatch"
    assert Y.shape == Yp.shape == Yr.shape, "Y/Y_pred/Y_res shape mismatch"
    assert y_mask_o.shape == y_mask_p.shape == y_mask_r.shape, "y_mask shape mismatch"
    print("OK")

for s in ["train","val","test"]:
=======
import numpy as np, os

def check(split, orig_dir="seq_ecef", dr_dir="dr_results"):
    o = np.load(os.path.join(orig_dir, f"{split}.npz"))
    p = np.load(os.path.join(dr_dir, f"{split}_Y_pred.npz"))
    r = np.load(os.path.join(dr_dir, f"{split}_Y_res.npz"))

    X, Y, y_mask_o = o["X"], o["Y"], o["y_mask"]
    Yp, y_mask_p   = p["Y_pred"], p["y_mask"]
    Yr, y_mask_r   = r["Y_res"],  r["y_mask"]

    print(f"\n[{split}]")
    print("N (X,Y):", X.shape[0], Y.shape[0])
    print("N (Y_pred):", Yp.shape[0], "N (Y_res):", Yr.shape[0])
    print("Y shapes:", Y.shape, Yp.shape, Yr.shape)
    print("mask shapes:", y_mask_o.shape, y_mask_p.shape, y_mask_r.shape)

    assert X.shape[0] == Y.shape[0] == Yp.shape[0] == Yr.shape[0], "Sequence count mismatch"
    assert Y.shape == Yp.shape == Yr.shape, "Y/Y_pred/Y_res shape mismatch"
    assert y_mask_o.shape == y_mask_p.shape == y_mask_r.shape, "y_mask shape mismatch"
    print("OK")

for s in ["train","val","test"]:
>>>>>>> c7f2eb60217fbb85e9d28131eb8c4c88577fb894
    check(s, orig_dir="data/seq_ecef", dr_dir="data/dr_results") 