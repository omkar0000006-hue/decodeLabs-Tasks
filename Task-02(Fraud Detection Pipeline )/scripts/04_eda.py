import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

INPUT_PATH = "data/processed/missing_values_handled.csv"

OUTPUT_DIR = "output/charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Dataset

df = pd.read_csv(INPUT_PATH)

# Dataset Information

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

print("\nShape :", df.shape)

print("\nColumns")

print(df.columns.tolist())

# Descriptive Statistics

print("\nSummary Statistics")

print(df.describe(include="all"))

# Numerical Columns

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Histograms

for col in numeric_cols:

    plt.figure(figsize=(6,4))

    sns.histplot(df[col], kde=True)

    plt.title(col)

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{col}_histogram.png")

    plt.close()

# Boxplots

for col in numeric_cols:

    plt.figure(figsize=(6,4))

    sns.boxplot(x=df[col])

    plt.title(col)

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{col}_boxplot.png")

    plt.close()

# Correlation Heatmap

plt.figure(figsize=(12,8))

sns.heatmap(

    df[numeric_cols].corr(),

    annot=True,

    cmap="coolwarm"

)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")

plt.close()

# Product Distribution

plt.figure(figsize=(10,5))

sns.countplot(

    data=df,

    x="Product",

    order=df["Product"].value_counts().index

)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/product_distribution.png")

plt.close()

# Payment Method

plt.figure(figsize=(8,5))

sns.countplot(

    data=df,

    x="PaymentMethod",

    order=df["PaymentMethod"].value_counts().index

)

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/payment_method.png")

plt.close()

# Coupon Code

plt.figure(figsize=(8,5))

sns.countplot(

    data=df,

    x="CouponCode",

    order=df["CouponCode"].value_counts().index

)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/coupon_usage.png")

plt.close()

# Referral Source

plt.figure(figsize=(8,5))

sns.countplot(

    data=df,

    x="ReferralSource",

    order=df["ReferralSource"].value_counts().index

)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/referral_source.png")

plt.close()

# Order Status

plt.figure(figsize=(8,5))

sns.countplot(

    data=df,

    x="OrderStatus",

    order=df["OrderStatus"].value_counts().index

)

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/order_status.png")

plt.close()

# Quantity vs Total Price

plt.figure(figsize=(8,6))

sns.scatterplot(

    data=df,

    x="Quantity",

    y="TotalPrice"

)

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/quantity_vs_price.png")

plt.close()

# Monthly Sales

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(df["Date"])

    monthly = df.groupby(df["Date"].dt.month)["TotalPrice"].sum()

    plt.figure(figsize=(10,5))

    monthly.plot(marker="o")

    plt.title("Monthly Sales")

    plt.xlabel("Month")

    plt.ylabel("Sales")

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/monthly_sales.png")

    plt.close()

print("\nEDA Completed Successfully")

print(f"\nCharts saved to {OUTPUT_DIR}")