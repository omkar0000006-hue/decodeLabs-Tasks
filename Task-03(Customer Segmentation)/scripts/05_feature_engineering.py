import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/missing_values_handled.csv")
OUTPUT_PATH = Path("data/processed/feature_engineered_data.csv")

def feature_engineering():
    df = pd.read_csv(
        INPUT_PATH,
        encoding="latin1",
        low_memory=False
    )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek

    if "TotalPrice" in df.columns:
        df["TotalPrice"] = pd.to_numeric(
            df["TotalPrice"],
            errors="coerce"
        )

    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

    if "UnitPrice" in df.columns:
        df["UnitPrice"] = pd.to_numeric(
            df["UnitPrice"],
            errors="coerce"
        )

    if "TotalPrice" in df.columns and "Quantity" in df.columns:
        df["AverageOrderValue"] = (
            df["TotalPrice"] /
            df["Quantity"].replace(0, 1)
        )

    if "Quantity" in df.columns:
        df["LargeOrder"] = (
            df["Quantity"] >= df["Quantity"].median()
        ).astype(int)

    if "TotalPrice" in df.columns:
        df["HighValueOrder"] = (
            df["TotalPrice"] >= df["TotalPrice"].median()
        ).astype(int)

    if "CouponCode" in df.columns:
        df["CouponUsed"] = (
            df["CouponCode"].notna()
        ).astype(int)

    if "Date" in df.columns:
        df["WeekendOrder"] = (
            df["Date"].dt.dayofweek >= 5
        ).astype(int)

    if "CustomerID" in df.columns:
        customer_counts = df["CustomerID"].value_counts()
        df["CustomerOrderFrequency"] = (
            df["CustomerID"].map(customer_counts)
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature Engineering Completed")
    print("\nColumns After Feature Engineering")
    print(df.columns.tolist())

    print("\nDataset Shape :", df.shape)

    print("\nFeature Engineered Dataset Saved")
    print(OUTPUT_PATH)

    return df

if __name__ == "__main__":
    feature_engineering()
    