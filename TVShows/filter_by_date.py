import pandas as pd
import os

def filter_csv(input_file, output_file, threshold_date="2015-01-01"):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 1. Load the dataset
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    # 2. Convert timestamp to datetime objects
    # errors='coerce' will turn invalid dates into NaT (Not a Time)
    df['dt_temp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # 3. Filter the data
    # We keep rows where the date is 2015-01-01 or later
    # We also drop rows where the timestamp was missing (NaT)
    original_count = len(df)
    df_filtered = df[df['dt_temp'] >= threshold_date].copy()
    
    # 4. Cleanup and Save
    df_filtered = df_filtered.drop(columns=['dt_temp'])
    df_filtered.to_csv(output_file, index=False)
    
    removed_count = original_count - len(df_filtered)
    print(f"Success!")
    print(f"Original rows: {original_count}")
    print(f"Rows removed (pre-2015 or invalid): {removed_count}")
    print(f"Final rows: {len(df_filtered)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    # Change these filenames to match your local files
    INPUT = "tv_scripts_dataset.csv"
    OUTPUT = "tv_dataset_after2015.csv"
    
    filter_csv(INPUT, OUTPUT)