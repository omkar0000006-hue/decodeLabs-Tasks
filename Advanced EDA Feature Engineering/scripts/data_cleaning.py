import os
import pandas as pd

INPUT_PATH = "data/raw/Dataset for Data Analytics - Sheet1.csv"
OUTPUT_PATH = "data/processed/cleaned_data.csv"

os.makedirs("data/processed", exist_ok=True)


def load_data():
    return pd.read_csv(INPUT_PATH)


def clean_data(df):
    df = df.drop_duplicates()

    df["Date"] = pd.to_datetime(df["Date"])

    text_columns = [
        "OrderID",
        "CustomerID",
        "Product",
        "ShippingAddress",
        "PaymentMethod",
        "OrderStatus",
        "TrackingNumber",
        "CouponCode",
        "ReferralSource"
    ]

    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    return df


def save_data(df):
    df.to_csv(OUTPUT_PATH, index=False)


def main():
    df = load_data()

    print("\nShape Before Cleaning:", df.shape)

    df = clean_data(df)

    print("Shape After Cleaning:", df.shape)

    print("\nData Types\n")
    print(df.dtypes)

    save_data(df)

    print("\nCleaned dataset saved successfully.")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()