import os
import pandas as pd

# Paths
input_root = r"E:\\dissertation\data\\all"
output_dir = r"E:\\dissertation\\cleaned_data"
os.makedirs(output_dir, exist_ok=True)

required_columns = [
    "time", "icao24", "lat", "lon", "velocity", "heading", "vertrate", 
    "baroaltitude", "hour"
]

# Columns to always drop (if they exist)
drop_if_exists = [
    "callsign", "onground", "alert", "spi", "squawk", 
    "lastposupdate", "lastcontact", "serials", "geoaltitude"
]

# Helper to count missing percentage
def missing_pct(series):
    return series.isnull().sum() / len(series)

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith(".csv"):
            filepath = os.path.join(root, file)
            print(f"📂 Processing: {filepath}")

            try:
                df = pd.read_csv(filepath)
                df.drop(columns=[col for col in drop_if_exists if col in df.columns], inplace=True)
                keep_cols = [col for col in required_columns if col in df.columns]
                df = df[keep_cols]
                
                for critical in ["time", "lat", "lon", "baroaltitude"]:
                    if critical in df.columns:
                        df = df[df[critical].notnull()]

                # For velocity-related columns: ffill then zero-fill
                for col in ["velocity", "heading", "vertrate"]:
                    if col in df.columns:
                        if missing_pct(df[col]) < 0.05:
                            df = df[df[col].notnull()]  # drop few missing
                        else:
                            df[col] = df[col].fillna(method="ffill").fillna(0)

                if "time" in df.columns:
                    df.sort_values("time", inplace=True)

                name_no_ext = os.path.splitext(file)[0]
                cleaned_filename = f"{name_no_ext}_cleaned.csv"
                output_path = os.path.join(output_dir, cleaned_filename)

                df.to_csv(output_path, index=False)
                print(f"✅ Saved cleaned file: {output_path}\n")

            except Exception as e:
                print(f"❌ Failed to process {file}: {e}")
