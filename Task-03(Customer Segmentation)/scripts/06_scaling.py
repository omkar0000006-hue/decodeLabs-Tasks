"""
06_scaling.py
---------------
Two steps combined:
  1. Scale  - StandardScaler on engineered features (equal weight
     before distance-based clustering).
  2. Compress (PCA) - reduce those scaled features to 2 principal
     components. Required by the brief so clustering runs on a
     compact, decorrelated space instead of 10 raw features, and so
     the cluster plot is interpretable.

Run:
    python 06_scaling.py

Output:
    data/processed/scaled_data.csv   -> standardized features
    data/processed/pca_data.csv      -> PC1, PC2 per row (used by clustering)
    models/scaler.pkl
    models/pca.pkl
    output/charts/pca_explained_variance.png
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

INPUT_PATH = Path("data/processed/feature_engineered_data.csv")
SCALED_OUTPUT_PATH = Path("data/processed/scaled_data.csv")
PCA_OUTPUT_PATH = Path("data/processed/pca_data.csv")
SCALER_PATH = Path("models/scaler.pkl")
PCA_MODEL_PATH = Path("models/pca.pkl")
VARIANCE_CHART_PATH = Path("output/charts/pca_explained_variance.png")

N_COMPONENTS = 2


def scale_and_reduce():
    df = pd.read_csv(INPUT_PATH, encoding="latin1", low_memory=False)

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    exclude_columns = ["Year", "Month", "Day", "DayOfWeek"]

    feature_columns = [c for c in numeric_columns if c not in exclude_columns]
    feature_columns = [c for c in feature_columns if df[c].notna().all()]

    X = df[feature_columns].copy()

    # ---- Step 1: Scale ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

    SCALED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    scaled_df.to_csv(SCALED_OUTPUT_PATH, index=False)
    joblib.dump({"scaler": scaler, "features": feature_columns}, SCALER_PATH)

    print("Scaling Completed")
    print("\nFeatures Used")
    print(feature_columns)
    print("\nScaled Shape :", scaled_df.shape)
    print("Scaled Dataset Saved ->", SCALED_OUTPUT_PATH)
    print("Scaler Saved ->", SCALER_PATH)

    # ---- Step 2: Compress (PCA) ----
    full_pca = PCA(n_components=len(feature_columns), random_state=42)
    full_pca.fit(scaled_df)
    cumulative_variance = full_pca.explained_variance_ratio_.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    plt.axhline(y=0.95, color="orange", linestyle="--", label="95% threshold")
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    VARIANCE_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(VARIANCE_CHART_PATH)
    plt.close()

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    components = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(N_COMPONENTS)])

    pca_df.to_csv(PCA_OUTPUT_PATH, index=False)
    joblib.dump({"pca": pca, "features": feature_columns}, PCA_MODEL_PATH)

    print("\nPCA Completed")
    print(f"Reduced {len(feature_columns)} features -> {N_COMPONENTS} components")
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance retained:", round(pca.explained_variance_ratio_.sum(), 4))
    print("PCA Dataset Saved ->", PCA_OUTPUT_PATH)
    print("PCA Model Saved ->", PCA_MODEL_PATH)
    print("Variance chart ->", VARIANCE_CHART_PATH)

    return scaled_df, pca_df


if __name__ == "__main__":
    scale_and_reduce()
