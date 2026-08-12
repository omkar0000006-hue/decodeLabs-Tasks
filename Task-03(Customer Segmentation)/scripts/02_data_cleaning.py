import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/raw/customer_data.csv")
OUTPUT_PATH = Path("data/processed/cleaned_data.csv")

def clean_dataset():
    print("\nCleaning Customer Dataset...\n")

    df = pd.read_csv(INPUT_PATH)

    print("Original Shape :", df.shape)

    duplicate_count = df.duplicated().sum()
    print("Duplicate Rows :", duplicate_count)

    df = df.drop_duplicates()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Date" in df.columns:
        print("Invalid Dates :", df["Date"].isna().sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Cleaned Shape :", df.shape)
    print("\nCleaned Dataset Saved")
    print(OUTPUT_PATH)

    return df

if __name__ == "__main__":
    clean_dataset()