import os
import logging
import pandas as pd

# Logging Configuration

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# File Paths

INPUT_PATH = "data/processed/cleaned_data.csv"

OUTPUT_PATH = "data/processed/missing_values_handled.csv"

# Load Dataset

def load_data():

    df = pd.read_csv(INPUT_PATH)

    return df


# Missing Value Report

def missing_report(df):

    print("\n" + "=" * 60)
    print("MISSING VALUE REPORT")
    print("=" * 60)

    missing = df.isnull().sum()

    print(missing)

    print("\nTotal Missing Values :", missing.sum())


# Handle Missing Values

def handle_missing(df):

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Fill numeric columns using median
    for col in numeric_cols:

        if df[col].isnull().sum() > 0:

            median = df[col].median()

            df[col].fillna(median, inplace=True)

    # Fill categorical columns using mode
    for col in categorical_cols:

        if df[col].isnull().sum() > 0:

            mode = df[col].mode()[0]

            df[col].fillna(mode, inplace=True)

    logging.info("Missing values handled successfully.")

    return df


# Save Dataset

def save_data(df):

    df.to_csv(

        OUTPUT_PATH,

        index=False

    )

    print("\nDataset Saved Successfully")

    print(OUTPUT_PATH)


# Main

def main():

    print("\nLoading Dataset...\n")

    df = load_data()

    missing_report(df)

    df = handle_missing(df)

    print("\nAfter Handling Missing Values\n")

    missing_report(df)

    save_data(df)


if __name__ == "__main__":

    main()