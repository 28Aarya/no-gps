"""
Path evaluation utilities for DR residual correction models.
Handles ECEF to LLA conversion and trajectory analysis.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


def ecef_to_lla(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ECEF coordinates to LLA (Latitude, Longitude, Altitude).
    
    Args:
        x, y, z: ECEF coordinates in meters
        
    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis
    f = 1/298.257223563  # flattening
    e2 = 2*f - f*f  # first eccentricity squared
    
    # Calculate longitude
    lon = np.arctan2(y, x) * 180.0 / np.pi
    
    # Calculate latitude and altitude using iterative method
    p = np.sqrt(x*x + y*y)
    lat = np.arctan2(z, p * (1 - e2))
    
    # Iterative refinement for latitude
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
    
    lat = lat * 180.0 / np.pi
    return lat, lon, alt


def compute_ade(pred_paths: np.ndarray, true_paths: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute Average Displacement Error (ADE) in meters.
    
    Args:
        pred_paths: Predicted paths [N, H, 3] (ECEF)
        true_paths: True paths [N, H, 3] (ECEF) 
        mask: Valid timesteps [N, H]
        
    Returns:
        ADE in meters
    """
    # Compute L2 distance at each timestep
    diff = pred_paths - true_paths  # [N, H, 3]
    l2_per_timestep = np.linalg.norm(diff, axis=-1)  # [N, H]
    
    # Apply mask and compute average
    masked_l2 = l2_per_timestep * mask
    total_error = np.sum(masked_l2)
    total_timesteps = np.sum(mask)
    
    return total_error / max(total_timesteps, 1)


def compute_fde(pred_paths: np.ndarray, true_paths: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute Final Displacement Error (FDE) in meters.
    
    Args:
        pred_paths: Predicted paths [N, H, 3] (ECEF)
        true_paths: True paths [N, H, 3] (ECEF)
        mask: Valid timesteps [N, H]
        
    Returns:
        FDE in meters
    """
    # Find final valid timestep for each sequence
    final_timesteps = np.argmax(mask, axis=1)  # [N]
    
    # Get final positions
    batch_indices = np.arange(len(final_timesteps))
    pred_final = pred_paths[batch_indices, final_timesteps]  # [N, 3]
    true_final = true_paths[batch_indices, final_timesteps]  # [N, 3]
    
    # Compute L2 distance at final timestep
    diff = pred_final - true_final
    l2_final = np.linalg.norm(diff, axis=-1)  # [N]
    
    return np.mean(l2_final)


def evaluate_paths(model: nn.Module, loader, device: torch.device, 
                scalers: Optional[Dict] = None, save_plots: bool = True, 
                output_dir: str = "results/path_eval") -> Dict[str, float]:
    """
    Comprehensive path evaluation with ECEF to LLA conversion and trajectory analysis.
    
    Args:
        model: Trained residual correction model
        loader: DataLoader with test data
        device: PyTorch device
        scalers: Optional scalers for denormalization
        save_plots: Whether to save trajectory plots
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Collect all predictions and ground truth
    all_pred_paths = []
    all_true_paths = []
    all_dr_paths = []
    all_masks = []
    all_pred_lens = []
    
    with torch.no_grad():
        for x, y_pred, y_res, y_mask, pred_lengths in loader:
            x, y_pred, y_res, y_mask = x.to(device), y_pred.to(device), y_res.to(device), y_mask.to(device)
            
            # Get predicted residuals
            pred_residuals = model(x, y_pred)
            
            # Convert to numpy for analysis
            pred_residuals_np = pred_residuals.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            y_res_np = y_res.cpu().numpy()
            y_mask_np = y_mask.cpu().numpy()
            pred_lengths_np = pred_lengths.cpu().numpy()
            
            # Denormalize if scalers provided
            if scalers is not None:
                # Denormalize predictions and residuals
                pred_residuals_np = denormalize_tensor(pred_residuals_np, scalers.get('y_res'))
                y_pred_np = denormalize_tensor(y_pred_np, scalers.get('y_pred'))
                y_res_np = denormalize_tensor(y_res_np, scalers.get('y_res'))
            
            # Compute final paths
            pred_paths = y_pred_np + pred_residuals_np  # DR + transformer residuals
            true_paths = y_pred_np + y_res_np  # DR + true residuals
            
            all_pred_paths.append(pred_paths)
            all_true_paths.append(true_paths)
            all_dr_paths.append(y_pred_np)
            all_masks.append(y_mask_np)
            all_pred_lens.append(pred_lengths_np)
    
    # Concatenate all batches
    pred_paths = np.concatenate(all_pred_paths, axis=0)
    true_paths = np.concatenate(all_true_paths, axis=0)
    dr_paths = np.concatenate(all_dr_paths, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    pred_lens = np.concatenate(all_pred_lens, axis=0)
    
    # Compute metrics
    metrics = {}
    
    # ADE and FDE for different path comparisons
    metrics['ade_dr_vs_true'] = compute_ade(dr_paths, true_paths, masks)
    metrics['ade_pred_vs_true'] = compute_ade(pred_paths, true_paths, masks)
    metrics['fde_dr_vs_true'] = compute_fde(dr_paths, true_paths, masks)
    metrics['fde_pred_vs_true'] = compute_fde(pred_paths, true_paths, masks)
    
    # Improvement metrics
    metrics['ade_improvement'] = metrics['ade_dr_vs_true'] - metrics['ade_pred_vs_true']
    metrics['fde_improvement'] = metrics['fde_dr_vs_true'] - metrics['fde_pred_vs_true']
    metrics['ade_improvement_pct'] = (metrics['ade_improvement'] / metrics['ade_dr_vs_true']) * 100
    metrics['fde_improvement_pct'] = (metrics['fde_improvement'] / metrics['fde_dr_vs_true']) * 100
    
    # Per prediction length analysis
    unique_pred_lens = np.unique(pred_lens)
    for pred_len in unique_pred_lens:
        mask_len = (pred_lens == pred_len)
        if np.sum(mask_len) > 0:
            metrics[f'ade_pred_vs_true_len_{pred_len}'] = compute_ade(
                pred_paths[mask_len], true_paths[mask_len], masks[mask_len]
            )
            metrics[f'ade_dr_vs_true_len_{pred_len}'] = compute_ade(
                dr_paths[mask_len], true_paths[mask_len], masks[mask_len]
            )
    
    # Convert to LLA and compute geographic metrics
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert sample trajectories to LLA
        sample_indices = np.random.choice(len(pred_paths), min(10, len(pred_paths)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            # Get valid timesteps
            valid_mask = masks[idx].astype(bool)
            if not np.any(valid_mask):
                continue
                
            # Extract valid portions
            pred_valid = pred_paths[idx][valid_mask]
            true_valid = true_paths[idx][valid_mask]
            dr_valid = dr_paths[idx][valid_mask]
            
            # Convert to LLA
            pred_lat, pred_lon, pred_alt = ecef_to_lla(pred_valid[:, 0], pred_valid[:, 1], pred_valid[:, 2])
            true_lat, true_lon, true_alt = ecef_to_lla(true_valid[:, 0], true_valid[:, 1], true_valid[:, 2])
            dr_lat, dr_lon, dr_alt = ecef_to_lla(dr_valid[:, 0], dr_valid[:, 1], dr_valid[:, 2])
            
            # Plot trajectory - Side by side comparison
            plt.figure(figsize=(16, 10))
            time_steps = np.arange(len(true_alt))
            
            # True vs Predicted Latitude
            plt.subplot(2, 3, 1)
            plt.plot(time_steps, true_lat, 'g-', label='True', linewidth=2)
            plt.plot(time_steps, pred_lat, 'b--', label='DR+Transformer', linewidth=2)
            plt.xlabel('Time Step')
            plt.ylabel('Latitude (deg)')
            plt.title(f'Latitude: True vs Predicted {i+1}')
            plt.legend()
            plt.grid(True)
            
            # True vs Predicted Longitude
            plt.subplot(2, 3, 2)
            plt.plot(time_steps, true_lon, 'g-', label='True', linewidth=2)
            plt.plot(time_steps, pred_lon, 'b--', label='DR+Transformer', linewidth=2)
            plt.xlabel('Time Step')
            plt.ylabel('Longitude (deg)')
            plt.title(f'Longitude: True vs Predicted {i+1}')
            plt.legend()
            plt.grid(True)
            
            # True vs Predicted Altitude
            plt.subplot(2, 3, 3)
            plt.plot(time_steps, true_alt, 'g-', label='True', linewidth=2)
            plt.plot(time_steps, pred_alt, 'b--', label='DR+Transformer', linewidth=2)
            plt.xlabel('Time Step')
            plt.ylabel('Altitude (m)')
            plt.title(f'Altitude: True vs Predicted {i+1}')
            plt.legend()
            plt.grid(True)
            
            # 2D trajectory plot (lat/lon)
            plt.subplot(2, 3, 4)
            plt.plot(true_lon, true_lat, 'g-', label='True', linewidth=2)
            plt.plot(dr_lon, dr_lat, 'r--', label='DR', linewidth=2)
            plt.plot(pred_lon, pred_lat, 'b:', label='DR+Transformer', linewidth=2)
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            plt.title(f'2D Trajectory {i+1}')
            plt.legend()
            plt.grid(True)
            
            # Error over time
            plt.subplot(2, 3, 5)
            dr_error = np.linalg.norm(true_valid - dr_valid, axis=1)
            pred_error = np.linalg.norm(true_valid - pred_valid, axis=1)
            plt.plot(time_steps, dr_error, 'r-', label='DR Error', linewidth=2)
            plt.plot(time_steps, pred_error, 'b-', label='DR+Transformer Error', linewidth=2)
            plt.xlabel('Time Step')
            plt.ylabel('Error (m)')
            plt.title(f'Error Over Time {i+1}')
            plt.legend()
            plt.grid(True)
            
            # Error distribution
            plt.subplot(2, 3, 6)
            plt.hist(dr_error, bins=20, alpha=0.7, label='DR Error', color='red')
            plt.hist(pred_error, bins=20, alpha=0.7, label='DR+Transformer Error', color='blue')
            plt.xlabel('Error (m)')
            plt.ylabel('Frequency')
            plt.title(f'Error Distribution {i+1}')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'trajectory_analysis_{i+1}.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    return metrics


def denormalize_tensor(tensor: np.ndarray, scaler) -> np.ndarray:
    """Denormalize tensor using sklearn scaler."""
    if scaler is None:
        return tensor
    
    original_shape = tensor.shape
    tensor_flat = tensor.reshape(-1, tensor.shape[-1])
    denorm_flat = scaler.inverse_transform(tensor_flat)
    return denorm_flat.reshape(original_shape)


def print_metrics(metrics: Dict[str, float]):
    """Print evaluation metrics in a formatted way."""
    print("\n" + "="*60)
    print("PATH EVALUATION METRICS")
    print("="*60)
    
    print(f"ADE (Average Displacement Error):")
    print(f"  DR vs True:           {metrics['ade_dr_vs_true']:.3f} m")
    print(f"  DR+Transformer vs True: {metrics['ade_pred_vs_true']:.3f} m")
    print(f"  Improvement:          {metrics['ade_improvement']:.3f} m ({metrics['ade_improvement_pct']:.1f}%)")
    
    print(f"\nFDE (Final Displacement Error):")
    print(f"  DR vs True:           {metrics['fde_dr_vs_true']:.3f} m")
    print(f"  DR+Transformer vs True: {metrics['fde_pred_vs_true']:.3f} m")
    print(f"  Improvement:          {metrics['fde_improvement']:.3f} m ({metrics['fde_improvement_pct']:.1f}%)")
    
    # Per prediction length metrics
    print(f"\nPer Prediction Length ADE:")
    for key, value in metrics.items():
        if key.startswith('ade_') and 'len_' in key:
            pred_len = key.split('_')[-1]
            if 'dr_vs_true' in key:
                print(f"  Length {pred_len}: DR={value:.3f} m", end="")
            else:
                print(f", DR+Transformer={value:.3f} m")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Example usage
    print("Path evaluation utilities loaded.")
    print("Use evaluate_paths() function to evaluate your trained model.")
