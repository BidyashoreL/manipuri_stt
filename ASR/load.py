import datasets
import soundfile as sf
import pandas as pd
import glob
import io
import os

# Directory containing your benchmark parquet files
folder_path = r"C:\Users\Churchil Ningthoujam\Downloads\manipuri_stt-20260816T162024Z-1-001\manipuri_stt\benchmark"
parquet_files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))

print(f"-> Found {len(parquet_files)} parquet file(s). Loading dataset...")
dataset = datasets.Dataset.from_parquet(parquet_files)

# Disable automatic audio decoding to prevent codec errors
if "audio_filepath" in dataset.column_names:
    dataset = dataset.cast_column("audio_filepath", datasets.Audio(decode=False))
elif "audio" in dataset.column_names:
    dataset = dataset.cast_column("audio", datasets.Audio(decode=False))

output_folder = "manipuri_benchmark_extracted"
os.makedirs(output_folder, exist_ok=True)

labels = []
print(f"-> Extracting audio files into '{output_folder}'...")

for i, sample in enumerate(dataset):
    audio_data = sample.get("audio_filepath") or sample.get("audio")
    text = sample.get("text", "")
    
    filename = f"benchmark_sample_{i}.wav"
    filepath = os.path.join(output_folder, filename)
    
    if isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"] is not None:
        data, samplerate = sf.read(io.BytesIO(audio_data["bytes"]))
        sf.write(filepath, data, samplerate)
    elif isinstance(audio_data, dict) and "array" in audio_data:
        sf.write(filepath, audio_data["array"], audio_data["sampling_rate"])
        
    labels.append({"file_name": filename, "text": text})
    
    if (i + 1) % 100 == 0:
        print(f"   Processed {i + 1}/{len(dataset)} files...")

# Save all labels to CSV
df = pd.DataFrame(labels)
df.to_csv(os.path.join(output_folder, "labels.csv"), index=False, encoding='utf-8')

print("-> Done! All benchmark files extracted successfully.")