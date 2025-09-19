"""Preprocess segmented ADS-B flight data for Transformer training.

Functions:
- resample_flight(df_flight, interval=6, gap_thresh=30)
- lla_to_ecef(lat, lon, alt)
- enu_to_ecef_velocity(v_e, v_n, v_u, lat, lon)
- compute_track(lat, lon): compute ground track from consecutive positions
- track_to_enu_velocity(track, velocity, vertrate): convert track to ENU velocity
- process_all_flights(df, interval=5, gap_thresh=30, drop_lla=True)

Notes:
- Large gaps (> gap_thresh seconds) are flagged in 'gap_flag'. If no gaps are
present in the entire dataset, the returned DataFrame omits 'gap_flag'.
- Heading is interpolated using angular unwrapping to avoid wrap-around
artifacts.
- Converts LLA to ECEF coordinates and computes ECEF velocity components.
- Computes ground track from consecutive lat/lon positions using great-circle formula.
- Uses track-based velocity computation instead of heading-based for better accuracy.
- Encodes heading as sin/cos and computes speed for DR and Transformer models.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


#Configs

DEFAULT_INTERVAL = 3
DEFAULT_GAP_THRESH = 90

def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Ensure a pandas Series is timezone-aware UTC datetime."""
    if pd.api.types.is_datetime64_any_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="s", utc=True, errors="coerce")


def _to_epoch_seconds(dt_index: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    dt = pd.DatetimeIndex(dt_index) if not isinstance(dt_index, pd.DatetimeIndex) else dt_index
    return dt.view("int64").astype("float64") / 1e9


def _build_resampled_index(start_ts: pd.Timestamp, end_ts: pd.Timestamp, interval_seconds: int) -> pd.DatetimeIndex:
    if pd.isna(start_ts) or pd.isna(end_ts):
        return pd.DatetimeIndex([], tz="UTC")
    freq = f"{int(interval_seconds)}s"
    return pd.date_range(start=start_ts, end=end_ts, freq=freq, inclusive="both", tz="UTC")


#Interpolation

def _pchip_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, is_angle_deg: bool = False) -> np.ndarray:
    """Interpolate using PCHIP. If angle, unwrap in radians before interpolation."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(x_new, np.nan, dtype="float64")

    xu, yu = x[mask], y[mask]
    order = np.argsort(xu)
    xu, yu = xu[order], yu[order]
    xu, unique_idx = np.unique(xu, return_index=True)
    yu = yu[unique_idx]
    
    if xu.size < 2:
        return np.full_like(x_new, np.nan, dtype="float64")

    if is_angle_deg:
        yu_rad = np.deg2rad(yu)
        yu_unwrapped = np.unwrap(yu_rad)
        interp = PchipInterpolator(xu, yu_unwrapped, extrapolate=True)
        y_new_unwrapped = interp(x_new)
        y_new = np.rad2deg(y_new_unwrapped)
        y_new = (y_new % 360.0 + 360.0) % 360.0
        return y_new.astype("float64")

    interp = PchipInterpolator(xu, yu, extrapolate=True)
    return interp(x_new).astype("float64")


def _mark_gap_flags(resampled_times_sec: np.ndarray, orig_times_sec: np.ndarray, gap_thresh: float) -> np.ndarray:
    """Return 1 where resampled time falls strictly inside a large gap, else 0."""
    if len(orig_times_sec) < 2:
        return np.zeros_like(resampled_times_sec, dtype=np.int8)
    
    sort_idx = np.argsort(orig_times_sec)
    t = orig_times_sec[sort_idx]
    dt = np.diff(t)
    gap_mask = dt > float(gap_thresh)
    
    if not gap_mask.any():
        return np.zeros_like(resampled_times_sec, dtype=np.int8)
    
    gaps = [(t[i], t[i + 1]) for i in np.where(gap_mask)[0]]
    flags = np.zeros_like(resampled_times_sec, dtype=np.int8)
    
    for t0, t1 in gaps:
        inside = (resampled_times_sec > t0) & (resampled_times_sec < t1)
        flags[inside] = 1
    return flags


#Geodesy utils

WGS84_A = 6378137.0  # Semi-major axis [m]
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 1 - (WGS84_B ** 2) / (WGS84_A ** 2)


def _lla_to_ecef(lat_rad: np.ndarray, lon_rad: np.ndarray, alt_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def lla_to_ecef(lat: Iterable[float], lon: Iterable[float], alt: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic LLA to ECEF coordinates."""
    lat, lon, alt = map(lambda x: np.asarray(x, dtype="float64"), [lat, lon, alt])
    
    if not (lat.shape == lon.shape == alt.shape):
        raise ValueError("lat, lon, alt must have the same shape")
    
    if lat.size == 0:
        return np.empty(0, dtype="float64"), np.empty(0, dtype="float64"), np.empty(0, dtype="float64")
    
    lat_rad, lon_rad = np.deg2rad(lat), np.deg2rad(lon)
    x, y, z = _lla_to_ecef(lat_rad, lon_rad, alt)
    return x.astype("float64"), y.astype("float64"), z.astype("float64")


def enu_to_ecef_velocity(v_e: Iterable[float], v_n: Iterable[float], v_u: Iterable[float],
                        lat: Iterable[float], lon: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ENU velocity components to ECEF velocity components."""
    v_e, v_n, v_u, lat, lon = map(lambda x: np.asarray(x, dtype="float64"), [v_e, v_n, v_u, lat, lon])
    
    if not (v_e.shape == v_n.shape == v_u.shape == lat.shape == lon.shape):
        raise ValueError("All inputs must have the same shape")
    
    if v_e.size == 0:
        return np.full_like(v_e, np.nan, dtype="float64"), np.full_like(v_e, np.nan, dtype="float64"), np.full_like(v_e, np.nan, dtype="float64")
    
    lat_rad, lon_rad = np.deg2rad(lat), np.deg2rad(lon)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    # Apply rotation: v_ecef = R^T * v_enu
    v_x = -sin_lon * v_e - sin_lat * cos_lon * v_n + cos_lat * cos_lon * v_u
    v_y = cos_lon * v_e - sin_lat * sin_lon * v_n + cos_lat * sin_lon * v_u
    v_z = cos_lat * v_n + sin_lat * v_u
    
    return v_x.astype("float64"), v_y.astype("float64"), v_z.astype("float64")


def compute_track(lat: Iterable[float], lon: Iterable[float]) -> np.ndarray:
    """Compute ground track from consecutive lat/lon positions using great-circle formula."""
    lat, lon = map(lambda x: np.asarray(x, dtype="float64"), [lat, lon])
    
    if not (lat.shape == lon.shape):
        raise ValueError("lat and lon must have the same shape")
    
    if lat.size < 2:
        return np.full_like(lat, np.nan, dtype="float64")
    
    # Convert to radians
    lat_rad, lon_rad = np.deg2rad(lat), np.deg2rad(lon)
    
    # Compute track using great-circle formula
    dlambda = lon_rad[1:] - lon_rad[:-1]
    y = np.sin(dlambda) * np.cos(lat_rad[1:])
    x = (np.cos(lat_rad[:-1]) * np.sin(lat_rad[1:]) - 
         np.sin(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.cos(dlambda))
    
    track_rad = np.arctan2(y, x)
    track_deg = np.rad2deg(track_rad)
    
    # Convert to degrees clockwise from north (0-360)
    track_deg = (track_deg + 360.0) % 360.0
    
    # First sample is NaN since we need two points to compute track
    result = np.full_like(lat, np.nan, dtype="float64")
    result[1:] = track_deg
    
    return result


def smooth_track(track_deg: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply moving average smoothing to track to reduce noise."""
    if track_deg.size < window_size:
        return track_deg.copy()
    
    # Use pandas rolling mean for NaN-aware smoothing
    track_series = pd.Series(track_deg)
    smoothed = track_series.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values.astype("float64")


def track_to_enu_velocity(track_deg: Iterable[float], velocity: Iterable[float], 
                        vertrate: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert track, velocity, and vertical rate to ENU velocity components."""
    track_deg, velocity, vertrate = map(lambda x: np.asarray(x, dtype="float64"), [track_deg, velocity, vertrate])
    
    if not (track_deg.shape == velocity.shape == vertrate.shape):
        raise ValueError("All inputs must have the same shape")
    
    if track_deg.size == 0:
        return np.full_like(track_deg, np.nan, dtype="float64"), np.full_like(track_deg, np.nan, dtype="float64"), np.full_like(track_deg, np.nan, dtype="float64")
    
    valid_mask = np.isfinite(track_deg)
    v_e = np.full_like(track_deg, np.nan, dtype="float64")
    v_n = np.full_like(track_deg, np.nan, dtype="float64")
    
    if valid_mask.any():
        track_rad = np.deg2rad(track_deg[valid_mask])
        v_n[valid_mask] = velocity[valid_mask] * np.cos(track_rad)  # North component
        v_e[valid_mask] = velocity[valid_mask] * np.sin(track_rad)  # East component
    
    v_u = vertrate  # Vertical rate
    
    return v_e.astype("float64"), v_n.astype("float64"), v_u.astype("float64")


def compute_speed(v_x: Iterable[float], v_y: Iterable[float], v_z: Iterable[float]) -> np.ndarray:
    """Compute speed as the norm of the ECEF velocity vector."""
    v_x, v_y, v_z = map(lambda x: np.asarray(x, dtype="float64"), [v_x, v_y, v_z])
    
    if not (v_x.shape == v_y.shape == v_z.shape):
        raise ValueError("All inputs must have the same shape")
    
    speed = np.sqrt(v_x**2 + v_y**2 + v_z**2)
    return speed.astype("float64")


def encode_heading(heading_deg: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Encode heading as sin and cos components."""
    heading_deg = np.asarray(heading_deg, dtype="float64")
    
    if heading_deg.size == 0:
        return np.full_like(heading_deg, np.nan, dtype="float64"), np.full_like(heading_deg, np.nan, dtype="float64")
    
    heading_rad = np.deg2rad(heading_deg)
    heading_sin, heading_cos = np.sin(heading_rad), np.cos(heading_rad)
    
    return heading_sin.astype("float64"), heading_cos.astype("float64")


#Core preprocessing

_NUMERIC_FEATURES_DEFAULT = ["lat", "lon", "baroaltitude", "velocity", "heading", "vertrate"]


def resample_flight(df_flight: pd.DataFrame, interval: int = DEFAULT_INTERVAL, gap_thresh: int = DEFAULT_GAP_THRESH) -> pd.DataFrame:
    """Resample a single flight's time series with PCHIP interpolation and gap flags."""

    if "time" not in df_flight.columns:
        raise ValueError("DataFrame must contain 'time' column")

    required_cols = ["icao24", "flight_id", "time"]
    missing = [c for c in required_cols if c not in df_flight.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_flight.copy()
    df["time"] = _ensure_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"])

    if df.empty:
        return df

    resampled_index = _build_resampled_index(df["time"].iloc[0], df["time"].iloc[-1], interval)
    if len(resampled_index) == 0:
        return df.iloc[0:0].copy()

    t_orig_sec = _to_epoch_seconds(df["time"])
    t_new_sec = _to_epoch_seconds(resampled_index)
    gap_flags = _mark_gap_flags(t_new_sec, t_orig_sec, float(gap_thresh))
    any_gaps = bool(gap_flags.any())

    numeric_cols = [c for c in _NUMERIC_FEATURES_DEFAULT if c in df.columns]
    data = {
        "time": resampled_index,
        "icao24": pd.Series(df["icao24"].iloc[0], index=resampled_index),
        "flight_id": pd.Series(df["flight_id"].iloc[0], index=resampled_index),
    }

    for col in numeric_cols:
        y = pd.to_numeric(df[col], errors="coerce").astype("float64").values
        is_angle = col == "heading"
        y_new = _pchip_interpolate(t_orig_sec, y, t_new_sec, is_angle_deg=is_angle)
        data[col] = y_new

    out = pd.DataFrame(data)
    if any_gaps:
        out["gap_flag"] = gap_flags.astype("int8")
    return out


def process_all_flights(df: pd.DataFrame, interval: int = DEFAULT_INTERVAL, gap_thresh: int = DEFAULT_GAP_THRESH, drop_lla: bool = True) -> pd.DataFrame:
    """Process a multi-flight DataFrame: resample, flag gaps, convert to ECEF, compute velocity features."""
    if df.empty:
        return df.copy()
        
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain 'time' column")
    
    drop_extra = [c for c in ["hour", "datetime"] if c in df.columns]
    if drop_extra:
        df = df.drop(columns=drop_extra)

    required_cols = ["icao24", "flight_id", "time", "lat", "lon", "baroaltitude", "velocity", "heading", "vertrate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    flights, any_gaps_overall = [], False

    for flight_id, df_flight in df.groupby("flight_id", sort=False):
        # STEP 1: Resample 
        rs = resample_flight(df_flight, interval=interval, gap_thresh=gap_thresh)
        
        if "gap_flag" in rs.columns and rs["gap_flag"].any():
            any_gaps_overall = True

        # STEP 2: Convert features using resampled data
        lat_series = pd.to_numeric(rs.get("lat", pd.Series(np.nan, index=rs.index)), errors="coerce")
        lon_series = pd.to_numeric(rs.get("lon", pd.Series(np.nan, index=rs.index)), errors="coerce")
        alt_series = pd.to_numeric(rs.get("baroaltitude", pd.Series(np.nan, index=rs.index)), errors="coerce")
        velocity_series = pd.to_numeric(rs.get("velocity", pd.Series(np.nan, index=rs.index)), errors="coerce")
        heading_series = pd.to_numeric(rs.get("heading", pd.Series(np.nan, index=rs.index)), errors="coerce")
        vertrate_series = pd.to_numeric(rs.get("vertrate", pd.Series(np.nan, index=rs.index)), errors="coerce")

        # Convert LLA to ECEF coordinates
        x, y, z = lla_to_ecef(lat_series.values, lon_series.values, alt_series.values)

        # Compute track from consecutive lat/lon positions
        track_deg = compute_track(lat_series.values, lon_series.values)
        track_deg = smooth_track(track_deg, window_size=3)
        
        # Convert track + velocity + vertrate to ENU velocity, then to ECEF
        v_e, v_n, v_u = track_to_enu_velocity(track_deg, velocity_series.values, vertrate_series.values)
        v_x, v_y, v_z = enu_to_ecef_velocity(v_e, v_n, v_u, lat_series.values, lon_series.values)

        # Encode heading and compute speed
        heading_sin, heading_cos = encode_heading(heading_series.values)
        speed = compute_speed(v_x, v_y, v_z)

        # Add all computed features
        rs["x"], rs["y"], rs["z"] = x, y, z
        rs["v_x"], rs["v_y"], rs["v_z"] = v_x, v_y, v_z
        rs["heading_sin"], rs["heading_cos"] = heading_sin, heading_cos
        rs["velocity"] = speed

        if drop_lla:
            for c in ["lat", "lon", "baroaltitude", "velocity", "heading", "vertrate"]:
                if c in rs.columns:
                    rs = rs.drop(columns=c)

        # Reorder columns
        desired = ["icao24", "flight_id", "time", "x", "y", "z", "v_x", "v_y", "v_z", "heading_sin", "heading_cos", "velocity"]
        if "gap_flag" in rs.columns:
            desired.append("gap_flag")
        desired_present = [c for c in desired if c in rs.columns]
        rs = rs[desired_present]
        flights.append(rs)

    out = pd.concat(flights, axis=0, ignore_index=True)
    # Ensure gap_flag exists for downstream sequence generator
    if "gap_flag" not in out.columns:
        out["gap_flag"] = np.zeros(len(out), dtype=np.int8)
    return out


#CSV conveniences

def process_csv_folder(input_dir: str, output_dir: Optional[str] = None, interval: int = DEFAULT_INTERVAL, 
                    gap_thresh: int = DEFAULT_GAP_THRESH, drop_lla: bool = True, glob_pattern: str = "*.csv") -> List[str]:
    """Process all CSVs in a folder; save as <name>_ECEF.csv."""
    if output_dir is None:
        output_dir = input_dir
    if not os.path.isdir(input_dir):
        raise ValueError(f"input_dir does not exist: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    import glob
    in_paths = sorted(glob.glob(os.path.join(input_dir, glob_pattern)))
    outputs = []
    
    for in_path in in_paths:
        try:
            df = pd.read_csv(in_path)
            cleaned = process_all_flights(df, interval=interval, gap_thresh=gap_thresh, drop_lla=drop_lla)
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(output_dir, f"{base}_ECEF.csv")
            cleaned.to_csv(out_path, index=False)
            outputs.append(out_path)
        except Exception as exc:
            print(f"Failed to process {in_path}: {exc}")
    return outputs


def _process_single_csv(input_csv: str, output_csv: Optional[str] = None, interval: int = DEFAULT_INTERVAL, 
                    gap_thresh: int = DEFAULT_GAP_THRESH, drop_lla: bool = True) -> str:
    df = pd.read_csv(input_csv)
    cleaned = process_all_flights(df, interval=interval, gap_thresh=gap_thresh, drop_lla=drop_lla)
    if output_csv is None:
        base = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(os.path.dirname(input_csv), f"{base}_ECEF.csv")
    cleaned.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resample ADS-B flights and convert to ECEF with velocity features.")
    parser.add_argument("input", help="Input CSV file or directory with CSVs")
    parser.add_argument("--output", "-o", help="Output CSV file or directory. Defaults alongside input.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help=f"Resample interval in seconds (default {DEFAULT_INTERVAL})")
    parser.add_argument("--gap", type=int, default=DEFAULT_GAP_THRESH, help=f"Gap threshold in seconds (default {DEFAULT_GAP_THRESH})")
    parser.add_argument("--keep-lla", action="store_true", help="Keep original lat/lon columns in output")
    args = parser.parse_args()

    path = args.input
    if os.path.isdir(path):
        outs = process_csv_folder(
            input_dir=path,
            output_dir=args.output if args.output else None,
            interval=args.interval,
            gap_thresh=args.gap,
            drop_lla=not args.keep_lla,
        )
        print("Saved:")
        for p in outs:
            print(p)
    else:
        out = _process_single_csv(
            input_csv=path,
            output_csv=args.output if args.output else None,
            interval=args.interval,
            gap_thresh=args.gap,
            drop_lla=not args.keep_lla,
        )
        print(out)


