import os
import pandas as pd

# File Paths

INPUT_PATH = "data/processed/missing_values_handled.csv"

OUTPUT_PATH = "data/processed/fraud_dataset.csv"

os.makedirs("data/processed", exist_ok=True)

# Load Dataset

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully")

print("Shape :", df.shape)

# Fraud Rules

def generate_fraud(row):

    score = 0

    # High value transaction
    if row["TotalPrice"] > 2500:
        score += 1

    # Large quantity
    if row["Quantity"] >= 5:
        score += 1

    # Large cart
    if row["ItemsInCart"] >= 6:
        score += 1

    # Coupon Used
    if str(row["CouponCode"]).strip() not in ["0", "None", "", "No Coupon"]:
        score += 1

    # Online payment
    if str(row["PaymentMethod"]).lower() in [
        "credit card",
        "debit card",
        "upi",
        "wallet",
        "net banking"
    ]:
        score += 1

    # Social Media Referral
    if str(row["ReferralSource"]).lower() in [
        "facebook",
        "instagram",
        "telegram"
    ]:
        score += 1

    # High Risk Orders
    if score >= 5:
        return 1

    return 0

# Create Fraud Column

print("\nGenerating Fraud Labels...")

df["Fraud"] = df.apply(generate_fraud, axis=1)

# Report

print("\nFraud Distribution")

print(df["Fraud"].value_counts())

print("\nFraud Percentage")

print(round(df["Fraud"].value_counts(normalize=True)*100,2))

# Save Dataset

df.to_csv(

    OUTPUT_PATH,

    index=False

)

print("\nFraud Dataset Saved Successfully")

print(OUTPUT_PATH)