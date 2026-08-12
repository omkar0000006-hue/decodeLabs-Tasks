# Project 3: Customer Segmentation (Unsupervised Learning)

Segments customers from order data into behavioural groups using
K-Means clustering, with PCA for dimensionality reduction, data-driven
K selection via Elbow + Silhouette methods, and a Flask app for live
predictions.

## Setup

```bash
pip install -r requirements.txt
```

## Run the pipeline (in order, from the project root)

```bash
python scripts/01_load_data.py            # inspect raw data
python scripts/02_data_cleaning.py        # dedupe, parse dates
python scripts/03_missing_values.py       # impute missing values
python scripts/04_eda.py                  # distribution charts
python scripts/05_feature_engineering.py  # derived features
python scripts/06_scaling.py              # StandardScaler + PCA (2 components)
python scripts/07_elbow_method.py         # Elbow + Silhouette -> optimal K
python scripts/08_train_clustering.py     # final K-Means model
python scripts/09_evaluate_clusters.py    # silhouette score, PCA plot, personas
python scripts/10_save_model.py           # confirm model files saved
python scripts/11_predict.py              # CLI: predict a new order's segment
```

## Run the web app

```bash
python app.py
```

Open `http://127.0.0.1:5000` — enter order details, get back the
predicted cluster + business persona (e.g. "High-Value Loyalists").
Requires the pipeline (scripts 01-10) to have run at least once.

## Project structure

```
Customer Segmentation/
├── data/
│   ├── raw/customer_data.csv
│   └── processed/            # generated at each pipeline stage
├── models/
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── customer_segmentation_model.pkl
│   └── optimal_k.txt
├── output/
│   ├── charts/                # EDA, elbow, silhouette, PCA variance, clusters
│   └── reports/
│       ├── customer_personas.csv
│       └── customer_personas.md
├── scripts/                   # 01-11, run in numeric order
├── app.py                     # Flask prediction demo
├── requirements.txt
└── README.md
```

## Pipeline details (matches the brief)

1. **Scale + Compress** (`06_scaling.py`) — `StandardScaler` puts all
   features on equal footing, then PCA reduces the 10 engineered
   features to 2 principal components (~66% variance retained; full
   explained-variance chart saved for review).
2. **Cluster** (`07_elbow_method.py` + `08_train_clustering.py`) —
   K-Means is tested for K=2..10 on the PCA data; both inertia (Elbow)
   and Silhouette Score are computed per K. The K with the **highest
   silhouette score** is picked automatically (K=4 on this dataset,
   silhouette ≈ 0.47).
3. **Evaluate + Translate** (`09_evaluate_clusters.py`) — final
   silhouette score and PCA-space cluster plot, then cluster centroids
   are mapped back to interpretable business metrics (avg spend, order
   frequency, coupon usage) and auto-labeled into personas like
   "High-Value Loyalists" or "Deal-Driven Budget Shoppers", each with a
   suggested marketing action.
4. **Predict** (`11_predict.py` / `app.py`) — both take raw order
   details, compute the derived features automatically, and run them
   through Scale → PCA → K-Means → persona lookup.

## Note on the dataset

`customer_data.csv` is order-level transaction data (Quantity,
UnitPrice, TotalPrice, OrderStatus, etc.), not pre-aggregated
per-customer profiles. Clustering therefore segments **order/behaviour
patterns**, not unique customers directly — worth mentioning in your
report/viva if asked.
