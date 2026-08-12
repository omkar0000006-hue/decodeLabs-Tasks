"""
app.py
-------
Flask web app for the Customer Segmentation project. Lets you enter a
new order's details and get back the predicted customer segment +
business persona, using the trained Scale -> PCA -> K-Means pipeline.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Requires the pipeline to have already been run once (scripts 01-10)
so that models/scaler.pkl, models/pca.pkl,
models/customer_segmentation_model.pkl, and
output/reports/customer_personas.csv all exist.
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

MODEL_PATH = Path("models/customer_segmentation_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
PCA_PATH = Path("models/pca.pkl")
PERSONAS_PATH = Path("output/reports/customer_personas.csv")

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Customer Segmentation</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f4f6f8; margin: 0; padding: 40px; }
    .card { background: #fff; max-width: 520px; margin: 0 auto; padding: 32px;
            border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
    h1 { font-size: 22px; margin-bottom: 4px; }
    p.subtitle { color: #666; margin-top: 0; margin-bottom: 24px; }
    label { display: block; margin-top: 14px; font-size: 14px; color: #333; }
    input, select { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box;
                    border: 1px solid #ccc; border-radius: 6px; font-size: 14px; }
    button { margin-top: 22px; width: 100%; padding: 10px; background: #2563eb;
             color: #fff; border: none; border-radius: 6px; font-size: 15px; cursor: pointer; }
    button:hover { background: #1e4fc4; }
    .result { margin-top: 24px; padding: 16px; border-radius: 8px; background: #eef6ec;
               border: 1px solid #b7dfae; }
    .result h2 { margin: 0 0 8px 0; font-size: 18px; color: #1a7a1a; }
    .error { margin-top: 24px; padding: 16px; border-radius: 8px; background: #fdecea;
              border: 1px solid #f5b5ad; color: #a12b22; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Customer Segmentation</h1>
    <p class="subtitle">Predict which customer segment a new order belongs to.</p>

    <form method="POST">
      <label>Quantity</label>
      <input type="number" step="any" name="quantity" value="{{ form.quantity or '' }}" required>

      <label>Unit Price</label>
      <input type="number" step="any" name="unit_price" value="{{ form.unit_price or '' }}" required>

      <label>Items In Cart</label>
      <input type="number" step="any" name="items_in_cart" value="{{ form.items_in_cart or '' }}" required>

      <label>Coupon Used?</label>
      <select name="coupon_used">
        <option value="y" {{ 'selected' if form.coupon_used == 'y' else '' }}>Yes</option>
        <option value="n" {{ 'selected' if form.coupon_used == 'n' else '' }}>No</option>
      </select>

      <label>Weekend Order?</label>
      <select name="is_weekend">
        <option value="y" {{ 'selected' if form.is_weekend == 'y' else '' }}>Yes</option>
        <option value="n" {{ 'selected' if form.is_weekend == 'n' else '' }}>No</option>
      </select>

      <label>Customer Order Frequency (past orders count)</label>
      <input type="number" step="any" name="order_frequency" value="{{ form.order_frequency or '' }}" required>

      <button type="submit">Predict Segment</button>
    </form>

    {% if error %}
      <div class="error"><strong>Error:</strong> {{ error }}</div>
    {% endif %}

    {% if result %}
      <div class="result">
        <h2>Cluster {{ result.cluster }}: {{ result.persona }}</h2>
        <p>{{ result.action }}</p>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler_data = joblib.load(SCALER_PATH)
    pca_data = joblib.load(PCA_PATH)
    return model, scaler_data, pca_data


def build_features(quantity, unit_price, items_in_cart, coupon_used,
                    is_weekend, customer_order_frequency):
    """Recreate the same derived features used in 05_feature_engineering.py."""
    total_price = quantity * unit_price
    average_order_value = total_price / quantity if quantity else total_price

    return {
        "Quantity": quantity,
        "UnitPrice": unit_price,
        "ItemsInCart": items_in_cart,
        "TotalPrice": total_price,
        "AverageOrderValue": average_order_value,
        "LargeOrder": int(quantity >= 3),
        "HighValueOrder": int(total_price >= 500),
        "CouponUsed": int(coupon_used),
        "WeekendOrder": int(is_weekend),
        "CustomerOrderFrequency": customer_order_frequency,
    }


def predict_segment(raw_features: dict):
    model, scaler_data, pca_data = load_artifacts()
    scaler, scaler_features = scaler_data["scaler"], scaler_data["features"]
    pca = pca_data["pca"]

    input_df = pd.DataFrame([raw_features])[scaler_features]
    scaled = pd.DataFrame(scaler.transform(input_df), columns=scaler_features)
    reduced = pd.DataFrame(
        pca.transform(scaled), columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    cluster = int(model.predict(reduced)[0])

    persona_name, action = None, None
    if PERSONAS_PATH.exists():
        personas = pd.read_csv(PERSONAS_PATH)
        match = personas[personas["Cluster"] == cluster]
        if not match.empty:
            persona_name = match.iloc[0]["PersonaName"]
            action = match.iloc[0]["SuggestedAction"]

    return cluster, persona_name, action


@app.route("/", methods=["GET", "POST"])
def index():
    form_values = {}
    result = None
    error = None

    if request.method == "POST":
        form_values = request.form.to_dict()
        try:
            if not (MODEL_PATH.exists() and SCALER_PATH.exists() and PCA_PATH.exists()):
                raise FileNotFoundError(
                    "Model files not found. Run scripts 01-10 first to train the pipeline."
                )

            features = build_features(
                quantity=float(form_values["quantity"]),
                unit_price=float(form_values["unit_price"]),
                items_in_cart=float(form_values["items_in_cart"]),
                coupon_used=form_values.get("coupon_used") == "y",
                is_weekend=form_values.get("is_weekend") == "y",
                customer_order_frequency=float(form_values["order_frequency"]),
            )

            cluster, persona_name, action = predict_segment(features)
            result = {
                "cluster": cluster,
                "persona": persona_name or "Unnamed segment",
                "action": action or "Run 09_evaluate_clusters.py to generate persona labels.",
            }
        except Exception as exc:  # surfaced directly to the form, not a 500 page
            error = str(exc)

    return render_template_string(PAGE_TEMPLATE, form=form_values, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
