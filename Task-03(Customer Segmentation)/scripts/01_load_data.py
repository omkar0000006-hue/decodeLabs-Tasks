import pandas as pd
import os

DATA_PATH = "data/raw/customer_data.csv"


def load_dataset():

    print("\nLoading Customer Dataset...")

    if not os.path.exists(DATA_PATH):
        print("\nERROR: Dataset not found!")
        print(f"Expected location: {DATA_PATH}")
        return None

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 60)
    print("CUSTOMER DATASET SUMMARY")
    print("=" * 60)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names")
    print("-" * 30)

    for column in df.columns:
        print(column)

    print("\nData Types")
    print("-" * 30)
    print(df.dtypes)

    print("\nMissing Values")
    print("-" * 30)
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print("-" * 30)
    print(df.duplicated().sum())

    print("\nFirst Five Records")
    print("-" * 30)
    print(df.head())

    print("\nDataset Loaded Successfully.")

    return df


if __name__ == "__main__":
    load_dataset()
    