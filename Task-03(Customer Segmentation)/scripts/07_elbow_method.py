"""
07_elbow_method.py
--------------------
Runs the "K dilemma" diagnostics required by the brief: for a range of
K values, computes both:
  - Inertia (Elbow Method) - where adding clusters stops helping
  - Silhouette Score - how well-separated/cohesive the clusters are

K-Means CANNOT tell you the right K on its own; these two diagnostics
together justify the choice mathematically instead of guessing.

Runs on the PCA-reduced data (not raw scaled features).

Run:
    python 07_elbow_method.py

Output:
    output/charts/elbow_method.png       -> inertia vs K
    output/charts/silhouette_scores.png  -> silhouette vs K
    models/optimal_k.txt                 -> chosen K, used by the next step
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INPUT_PATH = Path("data/processed/pca_data.csv")
ELBOW_CHART_PATH = Path("output/charts/elbow_method.png")
SILHOUETTE_CHART_PATH = Path("output/charts/silhouette_scores.png")
OPTIMAL_K_PATH = Path("models/optimal_k.txt")

K_RANGE = range(2, 11)


def find_optimal_k():
    df = pd.read_csv(INPUT_PATH)

    inertias = []
    silhouette_scores = []

    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(df)

        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(df, labels))

    # --- Elbow chart ---
    plt.figure(figsize=(8, 5))
    plt.plot(list(K_RANGE), inertias, marker="o")
    plt.title("Elbow Method (Inertia / WCSS)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.xticks(list(K_RANGE))
    plt.grid(True)
    plt.tight_layout()
    ELBOW_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ELBOW_CHART_PATH)
    plt.close()

    # --- Silhouette chart ---
    plt.figure(figsize=(8, 5))
    plt.plot(list(K_RANGE), silhouette_scores, marker="o", color="darkorange")
    plt.title("Silhouette Score by K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(list(K_RANGE))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SILHOUETTE_CHART_PATH)
    plt.close()

    # Optimal K = highest silhouette score (data-driven, not eyeballed)
    best_k = list(K_RANGE)[silhouette_scores.index(max(silhouette_scores))]

    Path("models").mkdir(parents=True, exist_ok=True)
    with open(OPTIMAL_K_PATH, "w") as f:
        f.write(str(best_k))

    print("=" * 60)
    print("OPTIMAL K SELECTION")
    print("=" * 60)
    for k, inertia, sil in zip(K_RANGE, inertias, silhouette_scores):
        marker = "  <-- best" if k == best_k else ""
        print(f"K={k:2d}  Inertia={inertia:10.2f}  Silhouette={sil:.4f}{marker}")

    print(f"\nRecommended K (highest silhouette score): {best_k}")
    print(f"Saved to {OPTIMAL_K_PATH}")
    print(f"\nCharts saved:\n  {ELBOW_CHART_PATH}\n  {SILHOUETTE_CHART_PATH}")

    return best_k


if __name__ == "__main__":
    find_optimal_k()
