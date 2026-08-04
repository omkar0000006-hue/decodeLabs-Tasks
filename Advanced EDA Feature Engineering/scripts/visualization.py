import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "data/processed/feature_engineered_data.csv"

os.makedirs("output/charts", exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f"output/charts/{filename}", dpi=300)
    plt.close()


def sales_by_product(df):
    plt.figure(figsize=(10, 6))
    df.groupby("Product")["TotalPrice"].sum().sort_values(ascending=False).plot(kind="bar")
    plt.title("Revenue by Product")
    plt.xlabel("Product")
    plt.ylabel("Revenue")
    save_plot("sales_by_product.png")


def payment_method(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="PaymentMethod")
    plt.title("Payment Method Distribution")
    save_plot("payment_method_distribution.png")


def order_status(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="OrderStatus")
    plt.title("Order Status Distribution")
    save_plot("order_status_distribution.png")


def monthly_sales(df):
    monthly = df.groupby("Month")["TotalPrice"].sum().reindex([
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ])

    plt.figure(figsize=(12, 6))
    monthly.plot(marker="o")
    plt.title("Monthly Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    save_plot("monthly_revenue.png")


def referral_source(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="ReferralSource")
    plt.title("Referral Source")
    save_plot("referral_source.png")


def quantity_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Quantity"], bins=10)
    plt.title("Quantity Distribution")
    save_plot("quantity_distribution.png")


def total_price_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["TotalPrice"], bins=30)
    plt.title("Total Price Distribution")
    save_plot("total_price_distribution.png")


def correlation_heatmap(df):
    plt.figure(figsize=(8, 6))

    numeric = df.select_dtypes(include=["int64", "float64"])

    sns.heatmap(
        numeric.corr(),
        annot=True,
        cmap="Blues"
    )

    plt.title("Correlation Heatmap")

    save_plot("correlation_heatmap.png")


def discount_analysis(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="DiscountApplied")
    plt.title("Discount Applied")
    save_plot("discount_applied.png")


def high_value_orders(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="HighValueOrder")
    plt.title("High Value Orders")
    save_plot("high_value_orders.png")


def main():

    df = load_data()

    sales_by_product(df)
    payment_method(df)
    order_status(df)
    monthly_sales(df)
    referral_source(df)
    quantity_distribution(df)
    total_price_distribution(df)
    correlation_heatmap(df)
    discount_analysis(df)
    high_value_orders(df)

    print("\n10 charts generated successfully.")
    print("Location : output/charts/")


if __name__ == "__main__":
    main()