
from datasets import load_dataset

dataset_id = "SPRINGLab/IndicTTS_Manipuri"

print("Opening IndicTTS in streaming mode...")
ds = load_dataset(
    dataset_id,
    split="train",
    streaming=True,
)

print("Reading the first sample...")
sample = next(iter(ds))

print("Columns:", list(sample.keys()))
print("Text:", sample.get("text"))
print("Gender:", sample.get("gender"))

audio = sample.get("audio")
if audio:
    print("Audio sample rate:", audio.get("sampling_rate"))
    print(
        "Audio duration (seconds):",
        len(audio["array"]) / audio["sampling_rate"],
    )

print("\nStreaming inspection complete.")
