"""
Debug script to test if model can overfit a single batch.
This is the most critical test - if the model can't achieve near-zero loss
on one batch, there's a fundamental bug.
"""

import torch
import torch.nn as nn
import numpy as np
from dr_former.data import build_dataloaders
from dr_former.dual_transformer import DualStreamTransformer
from dr_former.D_res import DRResidualFormer


def debug_single_batch(model, loader, device, max_epochs=1000):
    """
    Test if model can overfit a single batch.
    Should achieve near-zero loss if implementation is correct.
    """
    print("="*60)
    print("SINGLE BATCH OVERFITTING TEST")
    print("="*60)
    
    # Get one batch
    x, y_pred, y_res, y_mask, pred_lengths = next(iter(loader))
    x, y_pred, y_res, y_mask = x.to(device), y_pred.to(device), y_res.to(device), y_mask.to(device)
    
    print(f"Batch shapes:")
    print(f"  x: {x.shape}")
    print(f"  y_pred: {y_pred.shape}")
    print(f"  y_res: {y_res.shape}")
    print(f"  y_mask: {y_mask.shape}")
    print(f"  pred_lengths: {pred_lengths.shape}")
    
    # Check data ranges
    print(f"\nData ranges:")
    print(f"  x: [{x.min().item():.3f}, {x.max().item():.3f}]")
    print(f"  y_pred: [{y_pred.min().item():.3f}, {y_pred.max().item():.3f}]")
    print(f"  y_res: [{y_res.min().item():.3f}, {y_res.max().item():.3f}]")
    print(f"  y_mask: [{y_mask.min().item():.3f}, {y_mask.max().item():.3f}]")
    
    # Check for NaN/Inf
    print(f"\nNaN/Inf check:")
    print(f"  x has NaN: {torch.isnan(x).any().item()}")
    print(f"  y_pred has NaN: {torch.isnan(y_pred).any().item()}")
    print(f"  y_res has NaN: {torch.isnan(y_res).any().item()}")
    print(f"  y_mask has NaN: {torch.isnan(y_mask).any().item()}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='none')
    
    print(f"\nStarting overfitting test...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Overfitting loop
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred_residuals = model(x, y_pred)
        
        # Check model output
        if torch.isnan(pred_residuals).any():
            print(f"ERROR: Model output contains NaN at epoch {epoch}")
            break
            
        # Compute loss
        loss_per_element = criterion(pred_residuals, y_res)
        masked_loss = loss_per_element * y_mask.unsqueeze(-1)
        loss = masked_loss.sum() / y_mask.sum()
        
        if torch.isnan(loss):
            print(f"ERROR: Loss is NaN at epoch {epoch}")
            break
            
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        if torch.isnan(torch.tensor(total_grad_norm)):
            print(f"ERROR: Gradients contain NaN at epoch {epoch}")
            break
            
        optimizer.step()
        
        # Print progress
        if epoch % 100 == 0 or epoch < 10:
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.8f}, Grad Norm = {total_grad_norm:.6f}")
            
        # Check if converged
        if loss.item() < 1e-6:
            print(f"SUCCESS: Achieved near-zero loss ({loss.item():.2e}) at epoch {epoch}")
            break
            
        # Check if loss is increasing (diverging)
        if epoch > 50 and loss.item() > 100:
            print(f"ERROR: Loss is diverging ({loss.item():.2e}) at epoch {epoch}")
            break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_residuals = model(x, y_pred)
        loss_per_element = criterion(pred_residuals, y_res)
        masked_loss = loss_per_element * y_mask.unsqueeze(-1)
        final_loss = masked_loss.sum() / y_mask.sum()
        
        # Compute prediction accuracy
        pred_paths = y_pred + pred_residuals
        true_paths = y_pred + y_res
        path_error = torch.norm(pred_paths - true_paths, dim=-1) * y_mask
        avg_path_error = path_error.sum() / y_mask.sum()
        
        print(f"\nFinal Results:")
        print(f"  Residual Loss: {final_loss.item():.8f}")
        print(f"  Path Error: {avg_path_error.item():.6f} m")
        print(f"  Max Path Error: {path_error.max().item():.6f} m")
        
        # Check if model learned anything
        if final_loss.item() < 0.01:
            print("✅ SUCCESS: Model can overfit single batch")
            return True
        else:
            print("❌ FAILURE: Model cannot overfit single batch")
            return False


def debug_data_pipeline(orig_dir, dr_out_dir, scaler_pkl):
    """Debug the data pipeline for issues."""
    print("="*60)
    print("DATA PIPELINE DEBUG")
    print("="*60)
    
    try:
        loaders, scalers = build_dataloaders(
            orig_dir=orig_dir,
            dr_out_dir=dr_out_dir,
            batch_size=2,  # Small batch for debugging
            normalize=True,
            scaler_pkl=scaler_pkl,
        )
        
        print("✅ Data loaders created successfully")
        
        # Get one batch
        x, y_pred, y_res, y_mask, pred_lengths = next(iter(loaders["train"]))
        
        print(f"Data shapes:")
        print(f"  x: {x.shape}")
        print(f"  y_pred: {y_pred.shape}")
        print(f"  y_res: {y_res.shape}")
        print(f"  y_mask: {y_mask.shape}")
        print(f"  pred_lengths: {pred_lengths.shape}")
        
        # Check data consistency
        print(f"\nData consistency checks:")
        print(f"  y_pred and y_res same shape: {y_pred.shape == y_res.shape}")
        print(f"  y_mask matches y_pred: {y_mask.shape == y_pred.shape[:2]}")
        print(f"  pred_lengths shape: {pred_lengths.shape}")
        
        # Check residual calculation
        print(f"\nResidual calculation check:")
        print(f"  y_res range: [{y_res.min().item():.3f}, {y_res.max().item():.3f}]")
        print(f"  y_res mean: {y_res.mean().item():.6f}")
        print(f"  y_res std: {y_res.std().item():.6f}")
        
        # Check if residuals make sense
        if y_res.abs().max().item() > 1000:
            print("⚠️  WARNING: Residuals are very large (>1000m)")
        if y_res.std().item() < 0.001:
            print("⚠️  WARNING: Residuals have very low variance")
            
        return loaders, scalers
        
    except Exception as e:
        print(f"❌ ERROR in data pipeline: {e}")
        return None, None


def main():
    """Main debugging function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug single batch overfitting")
    parser.add_argument("--orig_dir", required=True, help="dir with train.npz/val.npz/test.npz")
    parser.add_argument("--dr_out_dir", required=True, help="dir with *_Y_pred.npz and *_Y_res.npz")
    parser.add_argument("--scaler_pkl", default="data/dr_results/train_standard_scalers.pkl")
    parser.add_argument("--arch", choices=["dr_res", "dual_stream"], default="dual_stream")
    parser.add_argument("--mode", choices=["encoder", "inverted"], default="inverted")
    parser.add_argument("--target_indices", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Debug data pipeline
    loaders, scalers = debug_data_pipeline(args.orig_dir, args.dr_out_dir, args.scaler_pkl)
    if loaders is None:
        return
    
    # Get data dimensions
    xb, ypb, yrb, ymb, plb = next(iter(loaders["train"]))
    T, F = xb.shape[1], xb.shape[2]
    H, D = ypb.shape[1], ypb.shape[2]
    
    # Create model
    if args.arch == "dual_stream":
        model = DualStreamTransformer(
            mode="time_first" if args.mode == "encoder" else "inverted",
            num_blocks=2,  # Small model for debugging
            seq_len=T, horizon=H, num_features=F, out_dim=D,
            d_model=64, num_heads=4, d_ff=128,  # Small dimensions
            dropout=0.0,  # No dropout for debugging
            activation="relu",
            target_indices=(args.target_indices if args.mode == "inverted" else None),
        ).to(device)
    else:
        model = DRResidualFormer(
            mode=args.mode,
            seq_len=T, horizon=H, num_features=F, out_dim=D,
            d_model=64, n_heads=4, d_ff=128, n_layers=2,
            dropout=0.0, target_indices=None if args.mode == "encoder" else args.target_indices,
            activation="relu",
        ).to(device)
    
    print(f"\nModel created: {args.arch}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test single batch overfitting
    success = debug_single_batch(model, loaders["train"], device)
    
    if success:
        print("\n✅ DEBUGGING PASSED: Model can overfit single batch")
        print("The implementation appears correct. The issue may be:")
        print("1. Hyperparameter tuning needed")
        print("2. Data quality issues")
        print("3. Model capacity vs dataset size")
    else:
        print("\n❌ DEBUGGING FAILED: Model cannot overfit single batch")
        print("There is a fundamental bug in the implementation.")
        print("Check:")
        print("1. Data pipeline")
        print("2. Model architecture")
        print("3. Loss function")
        print("4. Training loop")


if __name__ == "__main__":
    main()
