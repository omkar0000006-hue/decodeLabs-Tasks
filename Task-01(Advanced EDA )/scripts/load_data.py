import pandas as pd
import os

DATA_PATH = "data/raw/Dataset for Data Analytics - Sheet1.csv"


def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)

    return df


def dataset_summary(df):

    print("\nDataset Preview\n")
    print(df.head())

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nSummary Statistics")
    print(df.describe(include="all"))


if __name__ == "__main__":

    dataframe = load_dataset()

    dataset_summary(dataframe)