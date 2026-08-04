import os
import pandas as pd

INPUT_PATH = "data/processed/feature_engineered_data.csv"

os.makedirs("output/reports", exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def generate_report(df):

    report = []

    report.append("ADVANCED EDA & FEATURE ENGINEERING REPORT")
    report.append("=" * 45)

    report.append(f"\nDataset Shape : {df.shape}")

    report.append(f"\nTotal Revenue : ${df['TotalPrice'].sum():,.2f}")

    report.append(f"Average Order Value : ${df['TotalPrice'].mean():,.2f}")

    report.append(f"Total Customers : {df['CustomerID'].nunique()}")

    report.append(f"Total Products : {df['Product'].nunique()}")

    report.append(f"Highest Order Value : ${df['TotalPrice'].max():,.2f}")

    report.append(f"Lowest Order Value : ${df['TotalPrice'].min():,.2f}")

    report.append(
        f"\nTop Selling Product : {df['Product'].value_counts().idxmax()}"
    )

    report.append(
        f"Most Used Payment Method : {df['PaymentMethod'].value_counts().idxmax()}"
    )

    report.append(
        f"Most Common Referral Source : {df['ReferralSource'].value_counts().idxmax()}"
    )

    report.append("\nOrder Status Distribution")

    status_counts = df["OrderStatus"].value_counts()

    for status, count in status_counts.items():
        report.append(f"{status} : {count}")

    report.append(
        f"\nHigh Value Orders : {(df['HighValueOrder'] == 'Yes').sum()}"
    )

    report.append(
        f"Orders Using Coupon : {(df['DiscountApplied'] == 'Yes').sum()}"
    )

    report.append(
        f"Orders Without Coupon : {(df['DiscountApplied'] == 'No').sum()}"
    )

    with open(
        "output/reports/project_insights.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(report))

    print("\n".join(report))

    print("\nActual Order Status Values:\n")
    print(df["OrderStatus"].value_counts())

    print("\nInsights saved successfully.")


def main():

    df = load_data()

    generate_report(df)


if __name__ == "__main__":
    main()