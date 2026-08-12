"""
09_evaluate_clusters.py
--------------------------
Two steps combined:
  1. Evaluate - final silhouette score + PCA-space cluster plot.
  2. Translate - map cluster centroids back to interpretable business
     metrics (avg spend, order frequency, coupon usage) and
     auto-label each cluster as a business persona with a suggested
     marketing action. This is the brief's final deliverable:
     "translate resulting clusters into actionable business Personas."

Run:
    python 09_evaluate_clusters.py

Output:
    output/charts/customer_clusters.png
    output/reports/customer_personas.csv
    output/reports/customer_personas.md
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score

PCA_CLUSTERED_PATH = Path("data/processed/clustered_customers.csv")
FULL_CLUSTERED_PATH = Path("data/processed/clustered_full.csv")
CLUSTER_CHART_PATH = Path("output/charts/customer_clusters.png")
PERSONAS_CSV_PATH = Path("output/reports/customer_personas.csv")
PERSONAS_MD_PATH = Path("output/reports/customer_personas.md")

PERSONA_FEATURES = [
    "TotalPrice",
    "AverageOrderValue",
    "Quantity",
    "CustomerOrderFrequency",
    "CouponUsed",
    "HighValueOrder",
]


# ---------------------------------------------------------------- #
# Step 1: Evaluate
# ---------------------------------------------------------------- #
def evaluate_clusters():
    df = pd.read_csv(PCA_CLUSTERED_PATH)

    feature_cols = [c for c in df.columns if c != "Cluster"]
    X = df[feature_cols]
    labels = df["Cluster"]

    score = silhouette_score(X, labels)

    print("=" * 60)
    print("CLUSTER EVALUATION")
    print("=" * 60)
    print("\nFinal Silhouette Score :", round(score, 4))
    print("(closer to +1 = well separated, near 0 = overlapping clusters)")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="tab10", alpha=0.7
    )
    plt.title(f"Customer Segments in PCA Space (Silhouette = {score:.3f})")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    CLUSTER_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CLUSTER_CHART_PATH)
    plt.close()

    print("\nCluster visualization saved ->", CLUSTER_CHART_PATH)
    return score


# ---------------------------------------------------------------- #
# Step 2: Translate to business personas
# ---------------------------------------------------------------- #
def label_persona(row, overall_avg):
    high_spend = row["TotalPrice"] >= overall_avg["TotalPrice"]
    high_freq = row["CustomerOrderFrequency"] >= overall_avg["CustomerOrderFrequency"]
    high_coupon = row["CouponUsed"] >= overall_avg["CouponUsed"]

    if high_spend and high_freq:
        return "High-Value Loyalists", "VIP perks, early access, loyalty rewards to retain them."
    if high_spend and not high_freq:
        return "Big-Ticket One-Timers", "Re-engagement campaigns, personalized follow-up offers."
    if not high_spend and high_freq:
        return "Frequent Bargain Shoppers", "Bundle deals, subscription/loyalty programs to increase basket size."
    if high_coupon:
        return "Deal-Driven Budget Shoppers", "Flash sales, coupon-triggered campaigns, referral incentives."
    return "Low-Engagement / At-Risk", "Win-back campaigns, discovery offers, reduce churn risk."


def generate_personas():
    df = pd.read_csv(FULL_CLUSTERED_PATH, encoding="latin1", low_memory=False)

    available_features = [c for c in PERSONA_FEATURES if c in df.columns]
    overall_avg = df[available_features].mean()

    cluster_summary = df.groupby("Cluster")[available_features].mean().round(2)
    cluster_sizes = df["Cluster"].value_counts().sort_index()

    rows = []
    for cluster_id, row in cluster_summary.iterrows():
        name, action = label_persona(row, overall_avg)
        rows.append({
            "Cluster": cluster_id,
            "PersonaName": name,
            "CustomerCount": cluster_sizes[cluster_id],
            **row.to_dict(),
            "SuggestedAction": action,
        })

    personas_df = pd.DataFrame(rows)

    PERSONAS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    personas_df.to_csv(PERSONAS_CSV_PATH, index=False)

    lines = ["# Customer Segmentation Personas\n"]
    for _, r in personas_df.iterrows():
        lines.append(f"## Cluster {int(r['Cluster'])}: {r['PersonaName']}")
        lines.append(f"- Customers in segment: {int(r['CustomerCount'])}")
        for feat in available_features:
            lines.append(f"- Avg {feat}: {r[feat]}")
        lines.append(f"- **Suggested action:** {r['SuggestedAction']}\n")

    PERSONAS_MD_PATH.write_text("\n".join(lines))

    print("\n" + "=" * 60)
    print("CUSTOMER PERSONAS")
    print("=" * 60)
    print(personas_df.to_string(index=False))
    print("\nSaved ->", PERSONAS_CSV_PATH)
    print("Saved ->", PERSONAS_MD_PATH)

    return personas_df


if __name__ == "__main__":
    evaluate_clusters()
    generate_personas()
