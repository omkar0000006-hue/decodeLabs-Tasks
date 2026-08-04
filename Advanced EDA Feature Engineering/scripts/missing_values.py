import os
import pandas as pd

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/missing_values_handled.csv"

os.makedirs("data/processed", exist_ok=True)


def load_data():
    return pd.read_csv(INPUT_PATH)


def missing_value_report(df):
    report = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Percentage": (df.isnull().sum() / len(df) * 100).round(2)
    })

    report = report[report["Missing Values"] > 0]

    print("\nMissing Value Report\n")
    print(report)

    return report


def handle_missing_values(df):
    df["CouponCode"] = df["CouponCode"].fillna("No Coupon")
    return df


def save_data(df, report):
    df.to_csv(OUTPUT_PATH, index=False)
    report.to_csv("output/reports/missing_value_report.csv")


def main():
    os.makedirs("output/reports", exist_ok=True)

    df = load_data()

    report = missing_value_report(df)

    df = handle_missing_values(df)

    print("\nRemaining Missing Values\n")
    print(df.isnull().sum())

    save_data(df, report)

    print("\nDataset saved successfully.")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()