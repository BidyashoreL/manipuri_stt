from datasets import load_dataset

dataset_id = "SPRINGLab/IndicTTS_Manipuri"

print("Downloading Manipuri IndicTTS...")
ds = load_dataset(dataset_id, split="train")

print("\nDownload complete.")
print("Records:", len(ds))
print("Columns:", ds.column_names)
print("\nFirst sample:")

sample = ds[0]
print("Text:", sample["text"])
print("Gender:", sample.get("gender"))

audio = sample["audio"]
print("Audio sample rate:", audio.get("sampling_rate"))
print("Audio duration (seconds):", len(audio["array"]) / audio["sampling_rate"])
print("\nDataset is available in the Hugging Face cache.")
