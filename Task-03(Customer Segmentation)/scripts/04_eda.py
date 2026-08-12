import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("data/processed/missing_values_handled.csv")
OUTPUT_DIR = Path("output/charts")

def perform_eda():
    df = pd.read_csv(
        INPUT_PATH,
        encoding="latin1",
        low_memory=False
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(df.info())

    print("\nShape :", df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nSummary Statistics")
    print(df.describe(include="all"))

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    for column in numeric_columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df[column].dropna(), bins=30)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{column}_distribution.png")
        plt.close()

    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        if df[column].nunique() <= 20:
            counts = df[column].value_counts().head(20)

            plt.figure(figsize=(10, 5))
            counts.plot(kind="bar")
            plt.title(f"{column} Distribution")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{column}_distribution.png")
            plt.close()

    print("\nEDA Completed Successfully")
    print("Charts saved to", OUTPUT_DIR)

if __name__ == "__main__":
    perform_eda()
    