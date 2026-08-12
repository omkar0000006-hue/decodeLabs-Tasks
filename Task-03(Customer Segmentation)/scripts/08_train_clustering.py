"""
08_train_clustering.py
------------------------
Trains the final K-Means model on the PCA-reduced data, using the K
chosen by 07_elbow_method.py (falls back to 4 if that step wasn't run).

Cluster labels are merged back onto the ORIGINAL (unscaled,
un-PCA'd) feature-engineered data too, so 09_evaluate_clusters.py can
describe each cluster in human terms (e.g. "avg income $86k") instead
of meaningless PCA-space numbers.

Run:
    python 08_train_clustering.py

Output:
    models/customer_segmentation_model.pkl
    data/processed/clustered_customers.csv   (PCA cols + Cluster)
    data/processed/clustered_full.csv        (original cols + Cluster)
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.cluster import KMeans

PCA_INPUT_PATH = Path("data/processed/pca_data.csv")
ORIGINAL_INPUT_PATH = Path("data/processed/feature_engineered_data.csv")
OPTIMAL_K_PATH = Path("models/optimal_k.txt")
MODEL_PATH = Path("models/customer_segmentation_model.pkl")
CLUSTERED_PCA_PATH = Path("data/processed/clustered_customers.csv")
CLUSTERED_FULL_PATH = Path("data/processed/clustered_full.csv")

DEFAULT_K = 4


def get_optimal_k():
    if OPTIMAL_K_PATH.exists():
        return int(OPTIMAL_K_PATH.read_text().strip())
    print(f"optimal_k.txt not found, run 08_elbow_method.py first. Using default K={DEFAULT_K}.")
    return DEFAULT_K


def train_model():
    pca_df = pd.read_csv(PCA_INPUT_PATH)
    original_df = pd.read_csv(ORIGINAL_INPUT_PATH, encoding="latin1", low_memory=False)

    n_clusters = get_optimal_k()

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(pca_df)

    pca_df["Cluster"] = clusters

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    pca_df.to_csv(CLUSTERED_PCA_PATH, index=False)

    # Merge cluster labels back onto the original row order (same index)
    full_df = original_df.copy()
    full_df["Cluster"] = clusters
    full_df.to_csv(CLUSTERED_FULL_PATH, index=False)

    print("=" * 60)
    print("CUSTOMER SEGMENTATION")
    print("=" * 60)
    print(f"\nNumber of Clusters (K) : {n_clusters}")
    print("\nCluster Distribution")
    print(pca_df["Cluster"].value_counts().sort_index())
    print("\nModel saved ->", MODEL_PATH)
    print("Clustered PCA data saved ->", CLUSTERED_PCA_PATH)
    print("Clustered full data saved ->", CLUSTERED_FULL_PATH)


if __name__ == "__main__":
    train_model()
