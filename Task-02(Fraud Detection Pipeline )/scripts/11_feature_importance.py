import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Paths

DATA_PATH = "data/processed/feature_engineered_data.csv"

MODEL_PATH = "models/gradient_boosting.pkl"

OUTPUT_DIR = "output/charts"

REPORT_DIR = "output/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load Data

df = pd.read_csv(DATA_PATH)

X = df.drop("Fraud", axis=1)

# Load Model

model = joblib.load(MODEL_PATH)

# Feature Importance

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop Features\n")

print(importance_df)

# Save CSV

importance_df.to_csv(

    f"{REPORT_DIR}/feature_importance.csv",

    index=False

)

# Plot

plt.figure(figsize=(10,8))

plt.barh(

    importance_df["Feature"],

    importance_df["Importance"]

)

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.title("Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(

    f"{OUTPUT_DIR}/feature_importance.png"

)

plt.close()

print("\nFeature Importance Saved")

print("output/charts/feature_importance.png")

print("output/reports/feature_importance.csv")
