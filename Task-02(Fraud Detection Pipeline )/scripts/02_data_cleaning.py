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

INPUT_PATH = "data/raw/Dataset for Data Analytics - Sheet1.csv"
OUTPUT_PATH = "data/processed/cleaned_data.csv"

os.makedirs("data/processed", exist_ok=True)


# Load Dataset

def load_data():

    extension = INPUT_PATH.split(".")[-1].lower()

    if extension == "csv":
        df = pd.read_csv(INPUT_PATH)

    else:
        df = pd.read_excel(INPUT_PATH)

    return df


# Clean Dataset

def clean_data(df):

    print("\nCleaning Dataset...\n")

    print(f"Original Shape : {df.shape}")

    # Remove Duplicate Records

    duplicates = df.duplicated().sum()

    print(f"\nDuplicate Rows : {duplicates}")

    df = df.drop_duplicates()

    # Remove Leading / Trailing Spaces

    for col in df.select_dtypes(include="object").columns:

        df[col] = df[col].astype(str).str.strip()

    # Convert Date Column

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

    # Remove Rows with Invalid Dates

    invalid_dates = df["Date"].isna().sum()

    print(f"Invalid Dates : {invalid_dates}")

    df = df.dropna(subset=["Date"])

    # Remove Negative Quantity

    if "Quantity" in df.columns:

        df = df[df["Quantity"] >= 0]

    # Remove Negative UnitPrice

    if "UnitPrice" in df.columns:

        df = df[df["UnitPrice"] >= 0]

    # Reset Index

    df.reset_index(
        drop=True,
        inplace=True
    )

    print(f"\nCleaned Shape : {df.shape}")

    logging.info("Dataset Cleaned Successfully")

    return df


# Save Dataset

def save_data(df):

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nCleaned Dataset Saved")

    print(OUTPUT_PATH)

    logging.info("Cleaned Dataset Saved")


# Main

def main():

    df = load_data()

    df = clean_data(df)

    save_data(df)


if __name__ == "__main__":

    main()