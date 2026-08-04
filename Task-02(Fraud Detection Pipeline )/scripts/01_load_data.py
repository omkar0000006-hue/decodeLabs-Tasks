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

RAW_DATA_PATH = "data/raw/Dataset for Data Analytics - Sheet1.csv"

# Load Dataset

def load_dataset(path):

    try:

        if not os.path.exists(path):

            raise FileNotFoundError(f"\nDataset not found:\n{path}")

        extension = path.split(".")[-1].lower()

        if extension == "csv":

            df = pd.read_csv(path)

        elif extension in ["xlsx", "xls"]:

            df = pd.read_excel(path)

        else:

            raise Exception("Unsupported File Format")

        logging.info("Dataset Loaded Successfully")

        return df

    except Exception as e:

        logging.error(str(e))

        print(e)

        return None


# Dataset Information

def dataset_summary(df):

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names\n")

    for column in df.columns:

        print(column)

    print("\nData Types\n")

    print(df.dtypes)

    print("\nMissing Values\n")

    print(df.isnull().sum())

    print("\nFirst Five Records\n")

    print(df.head())

    logging.info("Dataset Summary Generated")


# Main

def main():

    print("\nLoading Dataset...")

    df = load_dataset(RAW_DATA_PATH)

    if df is not None:

        dataset_summary(df)

        print("\nDataset Loaded Successfully.")

    else:

        print("\nDataset Loading Failed.")


if __name__ == "__main__":

    main()