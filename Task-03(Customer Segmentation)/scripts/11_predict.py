"""
11_predict.py
--------------
Predicts which customer segment a NEW order belongs to.

Does NOT ask the user to manually type derived features
(AverageOrderValue, CustomerOrderFrequency, etc.) - those are computed
automatically from the raw inputs, then run through the same
Scale -> PCA -> K-Means pipeline used in training, and finally mapped
to a human-readable persona name.

Run:
    python 11_predict.py
"""

import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/customer_segmentation_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
PCA_PATH = Path("models/pca.pkl")
PERSONAS_PATH = Path("output/reports/customer_personas.csv")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler_data = joblib.load(SCALER_PATH)
    pca_data = joblib.load(PCA_PATH)
    return model, scaler_data, pca_data


def build_features(quantity, unit_price, items_in_cart, coupon_used,
                    is_weekend, customer_order_frequency):
    """Recreate the same derived features used in 05_feature_engineering.py
    from a handful of raw inputs, so the user doesn't have to guess them."""
    total_price = quantity * unit_price
    average_order_value = total_price / quantity if quantity else total_price

    return {
        "Quantity": quantity,
        "UnitPrice": unit_price,
        "ItemsInCart": items_in_cart,
        "TotalPrice": total_price,
        "AverageOrderValue": average_order_value,
        "LargeOrder": int(quantity >= 3),        # rough heuristic default
        "HighValueOrder": int(total_price >= 500),  # rough heuristic default
        "CouponUsed": int(coupon_used),
        "WeekendOrder": int(is_weekend),
        "CustomerOrderFrequency": customer_order_frequency,
    }


def predict_segment(raw_features: dict):
    model, scaler_data, pca_data = load_artifacts()
    scaler, scaler_features = scaler_data["scaler"], scaler_data["features"]
    pca, pca_features = pca_data["pca"], pca_data["features"]

    input_df = pd.DataFrame([raw_features])[scaler_features]
    scaled = pd.DataFrame(scaler.transform(input_df), columns=scaler_features)
    reduced = pd.DataFrame(pca.transform(scaled), columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    cluster = int(model.predict(reduced)[0])

    persona_name = None
    if PERSONAS_PATH.exists():
        personas = pd.read_csv(PERSONAS_PATH)
        match = personas[personas["Cluster"] == cluster]
        if not match.empty:
            persona_name = match.iloc[0]["PersonaName"]

    return cluster, persona_name


def main():
    print("\nCUSTOMER SEGMENTATION PREDICTION\n")

    quantity = float(input("Quantity: "))
    unit_price = float(input("UnitPrice: "))
    items_in_cart = float(input("ItemsInCart: "))
    coupon_used = input("Coupon used? (y/n): ").strip().lower() == "y"
    is_weekend = input("Weekend order? (y/n): ").strip().lower() == "y"
    customer_order_frequency = float(input("CustomerOrderFrequency (past orders count): "))

    features = build_features(
        quantity, unit_price, items_in_cart,
        coupon_used, is_weekend, customer_order_frequency,
    )

    cluster, persona_name = predict_segment(features)

    print("\n# Predicted Customer Segment")
    print(f"Cluster : {cluster}")
    if persona_name:
        print(f"Persona : {persona_name}")
    else:
        print("Persona : (run 11_generate_personas.py first for a persona name)")


if __name__ == "__main__":
    main()
