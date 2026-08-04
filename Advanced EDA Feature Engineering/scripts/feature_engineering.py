import os
import pandas as pd

INPUT_PATH = "data/processed/outliers_removed.csv"
OUTPUT_PATH = "data/processed/feature_engineered_data.csv"

os.makedirs("data/processed", exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def create_features(df):

    df["Month"] = df["Date"].dt.month_name()

    df["DayOfWeek"] = df["Date"].dt.day_name()

    df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)

    df["AverageItemPrice"] = (
        df["TotalPrice"] / df["Quantity"]
    ).round(2)

    df["DiscountApplied"] = df["CouponCode"].apply(
        lambda x: "Yes" if x != "No Coupon" else "No"
    )

    median_price = df["TotalPrice"].median()

    df["HighValueOrder"] = df["TotalPrice"].apply(
        lambda x: "Yes" if x >= median_price else "No"
    )

    return df


def save_data(df):
    df.to_csv(OUTPUT_PATH, index=False)


def main():

    df = load_data()

    print("\nColumns Before Feature Engineering\n")
    print(df.columns.tolist())

    df = create_features(df)

    print("\nColumns After Feature Engineering\n")
    print(df.columns.tolist())

    save_data(df)

    print("\nFeature engineered dataset saved successfully.")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()