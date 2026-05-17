import pandas as pd
import numpy as np
import os

# Paths to the dataset files
BASE_PATH = "../MahaDhwani/dataflow_pipeline/languages/Telugu"
METADATA_CSV = os.path.join(BASE_PATH, "video_ids_metadata_Telugu.csv")
TITLES_CSV = os.path.join(BASE_PATH, "video_ids_title_Telugu.csv")

# Constants
TARGET_HOURS = 100
TARGET_SECONDS = TARGET_HOURS * 3600

def main():
    output_summary_csv = "selected_telugu_summary.csv"
    output_ids_file = "selected_telugu_ids.txt"

    if os.path.exists(output_summary_csv) and os.path.exists(output_ids_file):
        print(f"Output files {output_summary_csv} and {output_ids_file} already exist. Skipping extraction.")
        return

    print(f"Loading metadata from {METADATA_CSV}...")
    df_meta = pd.read_csv(METADATA_CSV)
    
    print(f"Loading titles from {TITLES_CSV} (using python engine)...")
    df_titles = pd.read_csv(TITLES_CSV, engine='python')
    
    # Merge titles if needed
    df = pd.merge(df_meta, df_titles, on='id', how='left')
    
    # Shuffle the data for diversity
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Select rows to reach target duration
    selected_rows = []
    current_duration = 0
    
    for _, row in df.iterrows():
        if current_duration >= TARGET_SECONDS:
            break
        
        # Don't pick extremely short videos or extremely long ones
        # If duration is missing, skip
        if pd.isna(row['duration(sec)']) or row['duration(sec)'] < 60 or row['duration(sec)'] > 3600:
            continue
            
        selected_rows.append(row)
        current_duration += row['duration(sec)']
    
    df_selected = pd.DataFrame(selected_rows)
    actual_hours = current_duration / 3600
    
    print(f"Selected {len(df_selected)} videos.")
    print(f"Total duration: {actual_hours:.2f} hours ({current_duration:.0f} seconds).")
    
    # Save selected IDs
    df_selected['id'].to_csv(output_ids_file, index=False, header=False)
    
    # Save a summary CSV for verification
    df_selected.to_csv(output_summary_csv, index=False)
    
    print(f"\nSaved IDs to: {output_ids_file}")
    print(f"Saved summary to: {output_summary_csv}")
    
    # Display domain distribution
    print("\nDomain distribution in selected sample:")
    print(df_selected['domain'].value_counts().head(10))

if __name__ == "__main__":
    main()
