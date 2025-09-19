"""
Quick data inspection script to check for data pipeline issues.
"""

import numpy as np
import torch
from dr_former.data import build_dataloaders


def inspect_data(orig_dir, dr_out_dir, scaler_pkl):
    """Inspect the data for potential issues."""
    print("="*60)
    print("DATA INSPECTION")
    print("="*60)
    
    try:
        # Load raw data
        import os
        train_orig = np.load(os.path.join(orig_dir, "train.npz"))
        train_pred = np.load(os.path.join(dr_out_dir, "train_Y_pred.npz"))
        train_res = np.load(os.path.join(dr_out_dir, "train_Y_res.npz"))
        
        print("Raw data shapes:")
        print(f"  X: {train_orig['X'].shape}")
        print(f"  Y_pred: {train_pred['Y_pred'].shape}")
        print(f"  Y_res: {train_res['Y_res'].shape}")
        print(f"  y_mask: {train_res['y_mask'].shape}")
        
        # Check data ranges
        print(f"\nRaw data ranges:")
        print(f"  X: [{train_orig['X'].min():.3f}, {train_orig['X'].max():.3f}]")
        print(f"  Y_pred: [{train_pred['Y_pred'].min():.3f}, {train_pred['Y_pred'].max():.3f}]")
        print(f"  Y_res: [{train_res['Y_res'].min():.3f}, {train_res['Y_res'].max():.3f}]")
        
        # Check for NaN/Inf
        print(f"\nNaN/Inf check:")
        print(f"  X has NaN: {np.isnan(train_orig['X']).any()}")
        print(f"  Y_pred has NaN: {np.isnan(train_pred['Y_pred']).any()}")
        print(f"  Y_res has NaN: {np.isnan(train_res['Y_res']).any()}")
        
        # Check residual statistics
        print(f"\nResidual statistics:")
        print(f"  Y_res mean: {train_res['Y_res'].mean():.6f}")
        print(f"  Y_res std: {train_res['Y_res'].std():.6f}")
        print(f"  Y_res abs max: {np.abs(train_res['Y_res']).max():.3f}")
        
        # Check if residuals are too small/large
        if np.abs(train_res['Y_res']).max() > 1000:
            print("⚠️  WARNING: Residuals are very large (>1000m)")
        if train_res['Y_res'].std() < 0.001:
            print("⚠️  WARNING: Residuals have very low variance")
            
        # Check data loaders
        print(f"\nTesting data loaders...")
        loaders, scalers = build_dataloaders(
            orig_dir=orig_dir,
            dr_out_dir=dr_out_dir,
            batch_size=4,
            normalize=True,
            scaler_pkl=scaler_pkl,
        )
        
        # Get one batch
        x, y_pred, y_res, y_mask, pred_lengths = next(iter(loaders["train"]))
        
        print(f"Normalized batch shapes:")
        print(f"  x: {x.shape}")
        print(f"  y_pred: {y_pred.shape}")
        print(f"  y_res: {y_res.shape}")
        print(f"  y_mask: {y_mask.shape}")
        
        print(f"\nNormalized data ranges:")
        print(f"  x: [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"  y_pred: [{y_pred.min().item():.3f}, {y_pred.max().item():.3f}]")
        print(f"  y_res: [{y_res.min().item():.3f}, {y_res.max().item():.3f}]")
        
        # Check if normalization worked
        if x.std().item() < 0.1:
            print("⚠️  WARNING: Input features have very low variance after normalization")
        if y_res.std().item() < 0.1:
            print("⚠️  WARNING: Residuals have very low variance after normalization")
            
        print("✅ Data inspection completed")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in data inspection: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect data pipeline")
    parser.add_argument("--orig_dir", required=True, help="dir with train.npz/val.npz/test.npz")
    parser.add_argument("--dr_out_dir", required=True, help="dir with *_Y_pred.npz and *_Y_res.npz")
    parser.add_argument("--scaler_pkl", default="data/dr_results/train_standard_scalers.pkl")
    args = parser.parse_args()
    
    inspect_data(args.orig_dir, args.dr_out_dir, args.scaler_pkl)
