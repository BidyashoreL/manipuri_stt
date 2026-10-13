import pandas as pd
import json
import os

# 1. Load the local configuration
with open('config.json', 'r') as f:
    config = json.load(f)

print("Config loaded successfully!")

# 2. Read the CSV (Crucial: encoding='utf-8' forces Python to read the script correctly)
try:
    df = pd.read_csv(config['csv_path'], encoding='utf-8')
    print(f"\nTotal samples found in CSV: {len(df)}")
    print("\n--- First 3 rows of ground truth (Checking Manipuri text) ---")
    print(df.head(3))
except Exception as e:
    print(f"Error loading CSV: {e}")

# 3. Verify the audio paths
if 'df' in locals():
    first_audio_file = df.iloc[0]['file']
    full_audio_path = os.path.join(config['audio_folder'], first_audio_file)
    
    print(f"\n--- Audio Path Verification ---")
    print(f"Checking first audio file path: {full_audio_path}")
    if os.path.exists(full_audio_path):
        print("Success: Audio file found on disk! Your paths are wired up perfectly.")
    else:
        print("Warning: Audio file NOT found. Check your audio_folder path in config.json.")