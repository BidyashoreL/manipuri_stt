from datasets import load_dataset

print("Loading Manipuri subset in streaming mode...")

ds = load_dataset(
    "ai4bharat/indicvoices_r",
    "Manipuri",
    split="train",
    streaming=True
)

print("Dataset loaded.")

first = next(iter(ds))

print("\n===== COLUMNS =====")
print(first.keys())

print("\n===== SAMPLE =====")
for key, value in first.items():
    if key == "audio":
        print(f"{key}: {type(value)}")
        if isinstance(value, dict):
            print("  audio keys:", value.keys())
    else:
        print(f"{key}: {repr(value)[:500]}")

print("\n===== TEXT FIELDS =====")

for key, value in first.items():
    if isinstance(value, str):
        print(f"\n{key}:")
        print(value)