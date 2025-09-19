# no-gps: Learning to Correct Dead-Reckoning without GPS

This repo builds a data pipeline and residual Transformer ("DR-Former") that learns to correct a physics-based Dead Reckoning (DR) trajectory forecast using only onboard kinematics and no GPS during prediction. It also includes iTransformer- and PatchTST-style baselines, plus a vanilla encoder.

- **Data engineering** lives in `data/` (CSV cleaning → resampling → ECEF/velocity/heading features) and `d_reckoning/seq_dr.py` (sequence generation to NPZ).
- **DR baseline + residual labels** come from `d_reckoning/dr.py` which generates per-horizon DR predictions (`Y_pred`) and residuals (`Y_res = Y_true - Y_pred`).
- **Modeling** is in `dr_former/` (our main residual model), `iTransformer/`, `PatchTST/`, and `vanilla_encoder/`.
- **Training** lives in `dr_former/trainer/train.py` or `train/` utilities.


## Quick start

1) Create a Python env and install deps (PyTorch per your CUDA):
```bash
pip install -r requirements.txt
# Or install manually: torch torchvision torchaudio (with CUDA), numpy, pandas, scipy, scikit-learn, joblib, matplotlib, pyyaml, tqdm
```

2) Prepare CSVs. Put your raw ADS-B-like CSVs (with columns like `time, icao24, lat, lon, velocity, heading, vertrate, baroaltitude`) into a folder.

3) Clean and segment into flights:
```bash
# Clean raw CSVs (drops unused columns, fills minor gaps)
python data/preprocess.py  # edit input/output paths inside if needed

# Segment into flights and filter outliers by IQR
python data/segment.py
```
The scripts save cleaned and segmented CSVs to the paths set inside the files. Adjust the input/output paths at the top of `data/preprocess.py` and `data/segment.py` for your environment.

4) Resample, compute features, convert to ECEF, and build model-ready features:
```bash
python data/resample_ecef.py --help
python data/resample_ecef.py "PATH/TO/segmented" --output "PATH/TO/segmented_ecef" --interval 3 --gap 90
```
This produces per-flight time series with:
- Position in ECEF: `x, y, z`
- Velocity in ECEF: `v_x, v_y, v_z`
- Encoded heading: `heading_sin, heading_cos`
- Optional `gap_flag`

5) Turn the per-row CSV into sequences (X past window, Y future horizon) and save NPZs:
```bash
python d_reckoning/seq_dr.py \
  --train_csv PATH/TO/segmented_ecef/train.csv \
  --val_csv   PATH/TO/segmented_ecef/val.csv \
  --test_csv  PATH/TO/segmented_ecef/test.csv \
  --out_dir   data/new \
  --past_window 60 \
  --pred_len 5,20,60,100,140 \
  --max_sequences_per_shard 10000
```
This writes `data/new/{train,val,test}.npz` (or sharded `.part***.npz`) with arrays:
- `X`: [N, T, F] features (default order in `d_reckoning/seq_dr.py:FEATURES`)
- `Y`: [N, H, D] targets (ECEF `x,y,z`)
- `y_mask`: [N, H] valid prediction mask
- `pred_len`: [N] actual horizon per sequence


## Generate DR baseline and residuals

Compute DR forecasts per sequence and the residual labels used for learning:
```bash
python d_reckoning/dr.py --orig_dir data/new --out_dir data/dr_out --dt 3.0 --batch_size 50000
```
Outputs to `data/dr_out/`:
- `{split}_Y_pred.npz: Y_pred, y_mask`
- `{split}_Y_res.npz:  Y_res,  y_mask` where `Y_res = Y_true - Y_pred`

You can sanity-check alignment:
```python
from data.dataloader import check_alignment_splits
check_alignment_splits("data/new", "data/dr_out")
```


## Dataloader that fuses Original + DR outputs

`data/dataloader.py` provides `build_dataloaders(orig_dir, dr_out_dir, ...)` which:
- Loads `X` from `orig_dir/{split}.npz`
- Loads `Y_pred` and `Y_res` from `dr_out_dir/{split}_Y_pred.npz` and `{split}_Y_res.npz`
- Optionally fits/loads `StandardScaler`s for `x`, `y_pred`, `y_res`
- Applies outlier filtering on residuals if desired
- Returns PyTorch `DataLoader`s that yield `(x, y_pred, y_res, y_mask, pred_len)`

Example usage (inside training):
```python
loaders, scalers = build_dataloaders(
    orig_dir="data/new",
    dr_out_dir="data/dr_out",
    batch_size=64,
    normalize=True,
    scaler_pkl="results/dr/train_standard_scalers.pkl",
)
```


## Main model: DRResidualFormer (dr_former/)

The residual model learns `y_res` to correct DR’s `y_pred`. Final forecast is `y_hat = y_pred + y_res_hat`.

- Entry points:
  - `dr_former/D_res.py:DRResidualFormer` (two modes)
  - `dr_former/dual_transformer.py:DualStreamTransformer` (interleaved dual-stream fusion)
- Embeddings/fusion/heads:
  - `dr_former/embedding.py`: time/inverted embeddings and mapping of DR predictions to tokens
  - `dr_former/fusion.py`: cross-attention fusion
  - `dr_former/encoder_stack.py`: encoder stack built on `iTransformer` attention layers
  - `dr_former/output_head.py`: residual heads for different modes

Shapes:
- Inputs: `x` [B, T, F], `y_pred` [B, H, D]
- Output: `y_res_hat` [B, H, D]

Modes:
- `encoder` (time tokens): tokens along time; DR predictions embedded along horizon
- `inverted` (feature tokens): tokens are features; output head selects target indices to form [H, D]

Training loop: see `dr_former/trainer/train.py` which builds loaders, creates model, and trains with masked losses (MSE/Huber/Tukey or weighted variants) from `data/dataloader.py`.

Start training the residual model:
```bash
python dr_former/trainer/train.py \
  --orig_dir data/new \
  --dr_out_dir data/dr_out \
  --scaler_pkl results/dr/train_standard_scalers.pkl \
  --ckpt results/dr_res/best.pt \
  --arch dual_stream \
  --mode encoder \
  --d_model 192 --n_heads 6 --d_ff 384 --n_layers 3 --dropout 0.3 --activation relu \
  --batch_size 64 --epochs 50 --lr 1e-4 --weight_decay 1e-2 \
  --loss weighted_mse --outlier_threshold 2.0 --outlier_weight 0.1 \
  --scheduler plateau --plateau_patience 3 --plateau_factor 0.5
```
This prints per-prediction-length validation metrics and saves the best checkpoint and logs into `results/dr_res/`.


## iTransformer backbone and how it connects

- `iTransformer/iTransformer.py:AircraftiTransformer` is used as the encoder stack building block via `dr_former/encoder_stack.py` which imports attention layers from `iTransformer/layers/`.
- We reuse iTransformer’s inverted embedding idea (features as tokens) and `FullAttention` as the attention primitive.

If you want a direct iTransformer baseline (predict positions without DR residual learning), see `iTransformer/iTransformer.py` usage. Our DR-Former uses the same attention family but learns residuals over DR.


## Other models we experimented with

- `PatchTST/`: a PatchTST-style baseline over multivariate sequences
- `vanilla_encoder/`: simple Transformer encoder + heads
- `models/model_single_step.py` and `models/model_multi_step.py`: additional baselines for comparison

These are not the primary path, but can be used to reproduce ablations.


## Configuration

`utils/config.yaml` contains an example config for a PatchTST experiment. For DR-Former we drive most args via CLI to `dr_former/trainer/train.py`. Adjust paths, batch size, model dims, and loss scheduler as needed.


## Data formats and feature conventions

- Feature order for `X` is defined in `d_reckoning/seq_dr.py:FEATURES` and includes `time, gap_flag, heading_sin, heading_cos, v_x, v_y, v_z, x, y, z` (in ECEF).
- Target order `Y` is `x, y, z` in ECEF.
- Horizons are variable per sequence; `y_mask` marks valid steps. Losses are masked accordingly.


## Reproducing end-to-end

1) Clean, segment, and resample to ECEF with features (as above).
2) Generate sequences NPZs to `data/new/` using `d_reckoning/seq_dr.py`.
3) Run DR baseline to get `data/dr_out/{split}_Y_pred.npz` and `{split}_Y_res.npz` using `d_reckoning/dr.py`.
4) Train DR-Former with `dr_former/trainer/train.py`.
5) Evaluate test loss and per-horizon breakdown (printed and saved under the run directory).

Optionally verify integrity:
```python
from data.dataloader import check_alignment_splits
check_alignment_splits("data/new", "data/dr_out")  # ensures Y ≈ Y_pred + Y_res over mask
```


## Project structure (selected)

- `data/`: cleaning, segmentation, resampling/ECEF, dataloader, losses/metrics helpers
- `d_reckoning/`: sequence generator and DR baseline producing `Y_pred` and `Y_res`
- `dr_former/`: residual Transformer (embeddings, fusion, encoder stack, heads, trainer)
- `iTransformer/`: attention layers and inverted embedding concepts used by `dr_former`
- `PatchTST/`, `vanilla_encoder/`, `models/`: alternative baselines
- `train/`: misc training utilities (optimizers, callbacks, metrics)


## Notes

- Some scripts contain example Windows paths; update to your environment.
- Resolve any merge markers you see in local files before running production experiments.
- Ensure consistent ECEF units (meters) and constant resample interval `--interval` matches DR `--dt` (e.g., 3 s).
- For long horizons, consider robust losses (weighted MSE/Huber) provided in `data/dataloader.py`.