import os
import pandas as pd
import numpy as np
from glob import glob

def angular_diff(h1, h2):
    return abs((h1 - h2 + 180) % 360 - 180)

def iqr_outlier_filter(df, columns=['lat', 'lon', 'baroaltitude', 'vertrate', 'velocity', 'heading']):
    """
    Filter outliers using IQR method within each flight segment.
    
    Args:
        df: DataFrame with flight segments (must have 'flight_id' column)
        columns: List of columns to apply IQR filtering to
    
    Returns:
        filtered_df: DataFrame with outliers removed
    """
    if 'flight_id' not in df.columns:
        print("    ⚠️  No flight_id column found, skipping IQR filtering")
        return df
    
    original_len = len(df)
    filtered_segments = []
    
    for flight_id in df['flight_id'].unique():
        segment = df[df['flight_id'] == flight_id].copy()
        segment_original_len = len(segment)
        
        # Apply IQR filtering to each column
        for col in columns:
            if col in segment.columns:
                Q1 = segment[col].quantile(0.25)
                Q3 = segment[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Remove outliers
                segment = segment[(segment[col] >= lower_bound) & (segment[col] <= upper_bound)]
        
        # Only keep segment if it still has enough points
        if len(segment) >= 5:  # Minimum 5 points per segment
            filtered_segments.append(segment)
            if len(segment) < segment_original_len:
                removed = segment_original_len - len(segment)
                print(f"    🧹 Flight {flight_id}: Removed {removed}/{segment_original_len} outliers ({removed/segment_original_len*100:.1f}%)")
        else:
            print(f"    ❌ Flight {flight_id}: Removed entire segment (too few points after filtering)")
    
    if filtered_segments:
        filtered_df = pd.concat(filtered_segments, ignore_index=True)
        total_removed = original_len - len(filtered_df)
        if total_removed > 0:
            print(f"    📊 Total outliers removed: {total_removed}/{original_len} ({total_removed/original_len*100:.1f}%)")
        return filtered_df
    else:
        print(f"    ⚠️  All segments filtered out!")
        return pd.DataFrame()

def segment_and_assign_flight_ids(df, time_col='time', heading_col='heading',
                                time_gap_thresh=7200, heading_thresh=90, min_segment_len=10):
    df = df.sort_values(time_col).reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df[time_col], unit='s', utc=True)

    flight_id = 0
    segment_start = 0
    flight_ids = []

    for i in range(1, len(df)):
        t_gap = df.at[i, time_col] - df.at[i - 1, time_col]
        h_gap = angular_diff(df.at[i, heading_col], df.at[i - 1, heading_col])

        if t_gap > time_gap_thresh and h_gap > heading_thresh:
            if i - segment_start >= min_segment_len:
                flight_id += 1
                flight_ids.extend([flight_id] * (i - segment_start))
            segment_start = i

    # Add last chunk
    if len(df) - segment_start >= min_segment_len:
        flight_id += 1
        flight_ids.extend([flight_id] * (len(df) - segment_start))

    df = df.iloc[:len(flight_ids)].copy()
    df['flight_id'] = flight_ids

    return df, flight_id

def process_all_csvs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    csv_files = glob(os.path.join(input_folder, "*.csv"))
    total_segments = 0

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        if df.shape[0] < 20:
            print(f"⏭️  Skipping {filename} (too few rows)")
            continue

        try:
            print(f"\n📄 Processing {filename}...")
            segmented_df, num_flights = segment_and_assign_flight_ids(df)
            print(f"✅ {filename}: {num_flights} flight segments created")
            
            if num_flights > 0:
                # Apply IQR outlier filtering to the segmented data
                print(f"🔧 Applying IQR outlier filtering...")
                filtered_df = iqr_outlier_filter(segmented_df)
                
                if len(filtered_df) > 0:
                    out_path = os.path.join(output_folder, filename)
                    filtered_df.to_csv(out_path, index=False)
                    print(f"📁 Saved filtered file → {out_path}")
                    print(f"📊 Final: {len(filtered_df)} points across {filtered_df['flight_id'].nunique()} segments")
                else:
                    print(f"⚠️  All data filtered out for {filename}")
                
                total_segments += filtered_df['flight_id'].nunique() if len(filtered_df) > 0 else 0

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"\n🚀 Total flight segments across all files: {total_segments}")

process_all_csvs(input_folder="E:\dissertation\cleaned_data", output_folder="E:\dissertation\data\segmented")
