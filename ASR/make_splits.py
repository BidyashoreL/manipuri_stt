import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "unified_ground_truth.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Run build_unified_corpus.py first.")
        return

    print("Loading unified corpus...")
    df = pd.read_csv(csv_path)
    print(f"Total samples available: {len(df)}")

    # Stratify split by source or script if needed, or do a standard random split (85% train, 10% val, 5% test)
    train_val, test_df = train_test_split(df, test_size=0.05, random_state=42)
    train_df, val_df = train_test_split(train_val, test_size=0.1058, random_state=42) # 0.1058 of 95% is roughly 10% total

    # Save splits
    train_df.to_csv(os.path.join(base_dir, "ASR", "train.csv"), index=False)
    val_df.to_csv(os.path.join(base_dir, "ASR", "val.csv"), index=False)
    test_df.to_csv(os.path.join(base_dir, "ASR", "test.csv"), index=False)

    print("\nSuccessfully created dataset splits:")
    print(f"  - Training Set:   {len(train_df)} rows -> ASR/train.csv")
    print(f"  - Validation Set: {len(val_df)} rows -> ASR/val.csv")
    print(f"  - Test Set:       {len(test_df)} rows -> ASR/test.csv")

if __name__ == "__main__":
    create_splits()