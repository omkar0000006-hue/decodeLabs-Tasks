import os
import pandas as pd

from imblearn.over_sampling import SMOTE

# File Paths

INPUT_PATH = "data/processed/feature_engineered_data.csv"

OUTPUT_PATH = "data/processed/smote_data.csv"

os.makedirs("data/processed", exist_ok=True)

# Load Dataset

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully")

print("Shape :", df.shape)

# Features and Target

X = df.drop("Fraud", axis=1)

y = df["Fraud"]

print("\nClass Distribution Before SMOTE\n")

print(y.value_counts())

# Apply SMOTE

smote = SMOTE(

    sampling_strategy="auto",

    random_state=42,

    k_neighbors=5

)

X_resampled, y_resampled = smote.fit_resample(X, y)

# Create New DataFrame

smote_df = pd.DataFrame(

    X_resampled,

    columns=X.columns

)

smote_df["Fraud"] = y_resampled

# Save Dataset

smote_df.to_csv(

    OUTPUT_PATH,

    index=False

)

print("\nClass Distribution After SMOTE\n")

print(smote_df["Fraud"].value_counts())

print("\nNew Dataset Shape")

print(smote_df.shape)

print("\nSMOTE Dataset Saved Successfully")

print(OUTPUT_PATH)