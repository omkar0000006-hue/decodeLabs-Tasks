import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/cleaned_data.csv")
OUTPUT_PATH = Path("data/processed/missing_values_handled.csv")

def handle_missing_values():
    print("\nLoading Dataset...\n")

    df = pd.read_csv(
        INPUT_PATH,
        encoding="latin1",
        low_memory=False
    )

    print("=" * 60)
    print("MISSING VALUE REPORT")
    print("=" * 60)

    print(df.isnull().sum())

    total_missing = df.isnull().sum().sum()

    print("\nTotal Missing Values :", total_missing)

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            mode = df[column].mode()

            if len(mode) > 0:
                df[column] = df[column].fillna(mode.iloc[0])
            else:
                df[column] = df[column].fillna("Unknown")

    print("\nAfter Handling Missing Values\n")
    print(df.isnull().sum())

    print("\nTotal Missing Values :", df.isnull().sum().sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDataset Saved Successfully")
    print(OUTPUT_PATH)

    return df

if __name__ == "__main__":
    handle_missing_values()