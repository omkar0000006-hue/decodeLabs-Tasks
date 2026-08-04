import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# File Paths

INPUT_PATH = "data/processed/fraud_dataset.csv"

OUTPUT_PATH = "data/processed/feature_engineered_data.csv"

os.makedirs("data/processed", exist_ok=True)

# Load Dataset

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully")

print("\nColumns Before Feature Engineering\n")

print(df.columns.tolist())

# Convert Date

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month

df["Day"] = df["Date"].dt.day

df["DayOfWeek"] = df["Date"].dt.dayofweek

# New Features

# Average price per item

df["PricePerItem"] = df["TotalPrice"] / df["Quantity"]

# Large Cart Indicator

df["LargeCart"] = (df["ItemsInCart"] >= 5).astype(int)

# High Value Order

df["HighValueOrder"] = (df["TotalPrice"] >= 2500).astype(int)

# Coupon Used

df["CouponUsed"] = (
    df["CouponCode"]
    .astype(str)
    .str.lower()
    .isin(["none", "0", ""])
    .map({True: 0, False: 1})
)

# Weekend Purchase

df["WeekendOrder"] = df["DayOfWeek"].isin([5, 6]).astype(int)

# Encode Categorical Variables

encoder = LabelEncoder()

categorical_columns = [

    "Product",

    "ShippingAddress",

    "PaymentMethod",

    "OrderStatus",

    "TrackingNumber",

    "CouponCode",

    "ReferralSource"

]

for col in categorical_columns:

    df[col] = encoder.fit_transform(df[col].astype(str))

# Drop Unnecessary Columns

drop_columns = [

    "OrderID",

    "CustomerID",

    "Date"

]

df.drop(columns=drop_columns, inplace=True)

# Save Dataset

df.to_csv(

    OUTPUT_PATH,

    index=False

)

print("\nColumns After Feature Engineering\n")

print(df.columns.tolist())

print("\nDataset Shape :", df.shape)

print("\nFeature Engineered Dataset Saved")

print(OUTPUT_PATH)