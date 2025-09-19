import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dr_former.data import (build_dataloaders, masked_mse, masked_huber, masked_tukey_biweight, 
                        masked_weighted_mse, masked_weighted_huber, masked_weighted_tukey_biweight,
                        compute_per_horizon_loss, compute_per_predlen_loss)
from dr_former.D_res import DRResidualFormer


def create_loss_fn(loss_type, **kwargs):
    """Create loss function with given type and parameters."""
    if loss_type == "mse":
        return lambda pred, target, mask: masked_mse(pred, target, mask)
    elif loss_type == "huber":
        delta = kwargs.get("huber_delta", 1.0)
        return lambda pred, target, mask: masked_huber(pred, target, mask, delta=delta)
    elif loss_type == "tukey":
        c = kwargs.get("tukey_c", 4.685)
        return lambda pred, target, mask: masked_tukey_biweight(pred, target, mask, c=c)
    elif loss_type == "weighted_mse":
        threshold = kwargs.get("outlier_threshold", 2.0)
        weight = kwargs.get("outlier_weight", 0.1)
        return lambda pred, target, mask: masked_weighted_mse(pred, target, mask, outlier_threshold=threshold, outlier_weight=weight)
    elif loss_type == "weighted_huber":
        delta = kwargs.get("huber_delta", 1.0)
        threshold = kwargs.get("outlier_threshold", 2.0)
        weight = kwargs.get("outlier_weight", 0.1)
        return lambda pred, target, mask: masked_weighted_huber(pred, target, mask, delta=delta, outlier_threshold=threshold, outlier_weight=weight)
    elif loss_type == "weighted_tukey":
        c = kwargs.get("tukey_c", 4.685)
        threshold = kwargs.get("outlier_threshold", 2.0)
        weight = kwargs.get("outlier_weight", 0.1)
        return lambda pred, target, mask: masked_weighted_tukey_biweight(pred, target, mask, c=c, outlier_threshold=threshold, outlier_weight=weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def evaluate(model, loader, device, loss_fn, compute_predlen_losses=False, loss_type="mse", loss_kwargs=None):
    model.eval()
    total, count = 0.0, 0
    predlen_losses_accum = {}
    
    with torch.no_grad():
        for x, y_pred, y_res, y_mask, pred_lengths in loader:
            x, y_pred, y_res, y_mask = x.to(device), y_pred.to(device), y_res.to(device), y_mask.to(device)
            pred_lengths = pred_lengths.to(device)
            out = model(x, y_pred)
            loss = loss_fn(out, y_res, y_mask)
            total += loss.item()
            count += 1
            
            if compute_predlen_losses:
                per_predlen = compute_per_predlen_loss(out, y_res, y_mask, pred_lengths, loss_type, **(loss_kwargs or {}))
                for pred_len, loss_val in per_predlen.items():
                    if pred_len not in predlen_losses_accum:
                        predlen_losses_accum[pred_len] = []
                    predlen_losses_accum[pred_len].append(loss_val)
    
    avg_loss = total / max(count, 1)
    if compute_predlen_losses and predlen_losses_accum:
        # Average losses for each prediction length
        predlen_losses_avg = {pred_len: np.mean(losses) for pred_len, losses in predlen_losses_accum.items()}
        return avg_loss, predlen_losses_avg
    return avg_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    loaders, scalers = build_dataloaders(
        orig_dir=args.orig_dir,
        dr_out_dir=args.dr_out_dir,
        batch_size=args.batch_size,
        normalize=True,
        scaler_pkl=args.scaler_pkl,
    )
    xb, ypb, yrb, ymb, plb = next(iter(loaders["train"]))
    T, F = xb.shape[1], xb.shape[2]
    H, D = ypb.shape[1], ypb.shape[2]

    if args.mode == "inverted" and args.target_indices is not None:
        tgt_idx = [int(i) for i in args.target_indices]
    else:
        tgt_idx = None if args.mode == "encoder" else [0, 1, 2]

    if args.arch == "dual_stream":
        from dr_former.dual_transformer import DualStreamTransformer
        model = DualStreamTransformer(
            mode="time_first" if args.mode == "encoder" else "inverted",
            num_blocks=args.n_layers,
            seq_len=T, horizon=H, num_features=F, out_dim=D,
            d_model=args.d_model, num_heads=args.n_heads, d_ff=args.d_ff,
            dropout=args.dropout, activation=args.activation,
            target_indices=(tgt_idx if args.mode == "inverted" else None),
        ).to(device)
    else:
        model = DRResidualFormer(
            mode=args.mode,
            seq_len=T, horizon=H, num_features=F, out_dim=D,
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff, n_layers=args.n_layers,
            dropout=args.dropout, target_indices=None if args.mode == "encoder" else tgt_idx,
            activation=args.activation,
        ).to(device)


    # Model summary
    print("model summary:")
    print(f"  arch={args.arch}  mode={args.mode}  seq_len={T}  horizon={H}  features={F}  out_dim={D}")
    print(f"  blocks={args.n_layers}  d_model={args.d_model}  n_heads={args.n_heads}  d_ff={args.d_ff}")
    print(f"  activation={args.activation}  dropout={args.dropout}")
    print(f"  optimizer=AdamW  lr={args.lr}  weight_decay={args.weight_decay}  scheduler={args.scheduler}")
    if args.arch == "dual_stream":
        dual_mode = "time_first" if args.mode == "encoder" else "inverted"
        print(f"  dual_stream mode={dual_mode}")
    if args.mode == "inverted":
        print(f"  target_indices={tgt_idx}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler
    scheduler = None
    if args.scheduler == "cosine":
        t_max = args.t_max if args.t_max and args.t_max > 0 else args.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=args.eta_min)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=args.plateau_patience, factor=args.plateau_factor, min_lr=args.plateau_min_lr
        )

    # Create loss function using centralized factory
    loss_fn = create_loss_fn(args.loss, 
                            huber_delta=args.huber_delta,
                            tukey_c=args.tukey_c,
                            outlier_threshold=args.outlier_threshold,
                            outlier_weight=args.outlier_weight)

    # Loss kwargs for per-horizon computation
    loss_kwargs = {}
    if "huber" in args.loss:
        loss_kwargs["delta"] = args.huber_delta
    if "tukey" in args.loss:
        loss_kwargs["c"] = args.tukey_c
    if "weighted" in args.loss:
        loss_kwargs["outlier_threshold"] = args.outlier_threshold
        loss_kwargs["outlier_weight"] = args.outlier_weight

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    log_rows = [("epoch", "train_loss", "val_loss", "lr")]
    horizon_log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for x, y_pred, y_res, y_mask, pred_lengths in loaders["train"]:
            x, y_pred, y_res, y_mask = x.to(device), y_pred.to(device), y_res.to(device), y_mask.to(device)
            out = model(x, y_pred)
            loss = loss_fn(out, y_res, y_mask)
            opt.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            running += loss.item()
            steps += 1

        train_loss = running / max(steps, 1)
        
        # Compute per-prediction-length losses every few epochs
        if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            val_loss, val_predlen_losses = evaluate(model, loaders["val"], device, loss_fn, 
                                                compute_predlen_losses=True, loss_type=args.loss, loss_kwargs=loss_kwargs)
            current_lr = opt.param_groups[0]["lr"]
            print(f"epoch {epoch:03d}  train {train_loss:.6f}  val {val_loss:.6f}  lr {current_lr:.6g}")
            # Show losses grouped by prediction lengths
            pred_len_strs = [f"len={pred_len}:{loss_val:.4f}" for pred_len, loss_val in sorted(val_predlen_losses.items())]
            print(f"  val per-pred-len: {' '.join(pred_len_strs)}")
            
            # Store for CSV logging (convert dict to list for consistency)
            sorted_pred_lens = sorted(val_predlen_losses.keys())
            horizon_log_rows.append([epoch] + [val_predlen_losses[pl] for pl in sorted_pred_lens])
        else:
            val_loss = evaluate(model, loaders["val"], device, loss_fn)
            current_lr = opt.param_groups[0]["lr"]
            print(f"epoch {epoch:03d}  train {train_loss:.6f}  val {val_loss:.6f}  lr {current_lr:.6g}")

        log_rows.append((epoch, train_loss, val_loss, current_lr))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": vars(args)}, args.ckpt)
            print(f"saved best -> {args.ckpt}")

        # step scheduler
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

    # save logs
    log_dir = os.path.dirname(args.ckpt)
    csv_path = os.path.join(log_dir, "loss_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(log_rows)
    
    # Save per-prediction-length losses
    if horizon_log_rows:
        # Get prediction lengths from first row (excluding epoch)
        first_epoch_losses = horizon_log_rows[0]
        if len(first_epoch_losses) > 1:  # has prediction length data
            # Infer prediction lengths from first detailed evaluation
            val_loss_sample, val_predlen_sample = evaluate(model, loaders["val"], device, loss_fn, 
                                                        compute_predlen_losses=True, loss_type=args.loss, loss_kwargs=loss_kwargs)
            sorted_pred_lens = sorted(val_predlen_sample.keys())
            predlen_csv_path = os.path.join(log_dir, "predlen_losses.csv")
            with open(predlen_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch"] + [f"pred_len_{pl}" for pl in sorted_pred_lens])
                w.writerows(horizon_log_rows)
    try:
        import matplotlib.pyplot as plt
        xs = [r[0] for r in log_rows[1:]]
        tr = [r[1] for r in log_rows[1:]]
        vl = [r[2] for r in log_rows[1:]]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, tr, label="train")
        plt.plot(xs, vl, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"plotting failed: {e}")

    # Final test evaluation with per-prediction-length breakdown
    test_loss, test_predlen_losses = evaluate(model, loaders["test"], device, loss_fn, 
                                            compute_predlen_losses=True, loss_type=args.loss, loss_kwargs=loss_kwargs)
    print(f"test {test_loss:.6f}")
    # Show losses grouped by prediction lengths
    test_pred_len_strs = [f"len={pred_len}:{loss_val:.4f}" for pred_len, loss_val in sorted(test_predlen_losses.items())]
    print(f"test per-pred-len: {' '.join(test_pred_len_strs)}")
    
    # Save test prediction length losses
    test_predlen_csv = os.path.join(log_dir, "test_predlen_losses.csv")
    with open(test_predlen_csv, "w", newline="") as f:
        w = csv.writer(f)
        sorted_pred_lens = sorted(test_predlen_losses.keys())
        w.writerow([f"pred_len_{pl}" for pl in sorted_pred_lens])
        w.writerow([test_predlen_losses[pl] for pl in sorted_pred_lens])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train residual model to correct DR predictions")
    p.add_argument("--orig_dir", required=True, help="dir with train.npz/val.npz/test.npz")
    p.add_argument("--dr_out_dir", required=True, help="dir with *_Y_pred.npz and *_Y_res.npz")
    p.add_argument("--scaler_pkl", default="results/dr/train_standard_scalers.pkl")
    p.add_argument("--ckpt", default="results/dr_res/best.pt")
    p.add_argument("--mode", choices=["encoder", "inverted"], default="encoder")
    p.add_argument("--target_indices", nargs="*", type=int, help="feature indices for targets (inverted mode)")
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--activation", choices=["gelu", "relu", "silu", "elu"], default="relu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--loss", choices=["mse", "huber", "tukey", "weighted_mse", "weighted_huber", "weighted_tukey"], default="weighted_mse")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--tukey_c", type=float, default=4.685)
    p.add_argument("--outlier_threshold", type=float, default=2.0, help="weighted_mse: residuals above this many std devs are outliers")
    p.add_argument("--outlier_weight", type=float, default=0.1, help="weighted_mse: weight multiplier for outliers")
    p.add_argument("--scheduler", choices=["none", "step", "cosine", "plateau"], default="plateau")
    p.add_argument("--step_size", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--t_max", type=int, default=0, help="CosineAnnealingLR T_max; 0 -> use epochs")
    p.add_argument("--eta_min", type=float, default=0.0)
    p.add_argument("--plateau_patience", type=int, default=3)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=0.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--arch", choices=["dr_res","dual_stream"], default="dual_stream", help="Model architecture to use")
    args = p.parse_args()
    train(args)
