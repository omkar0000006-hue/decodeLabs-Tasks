from flask import Flask, request, jsonify
import joblib
import pandas as pd


app = Flask(__name__)


# Load Model

model = joblib.load(
    "models/best_gradient_boosting.pkl"
)


@app.route("/")
def home():
    return "Fraud Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()


        df = pd.DataFrame([data])


        # Feature Engineering (same as training)

        df["PricePerItem"] = (
            df["TotalPrice"] / df["Quantity"]
        )


        df["LargeCart"] = (
            df["ItemsInCart"] > 10
        ).astype(int)


        df["HighValueOrder"] = (
            df["TotalPrice"] > 5000
        ).astype(int)


        df["CouponUsed"] = (
            df["CouponCode"] != 0
        ).astype(int)


        df["WeekendOrder"] = (
            df["DayOfWeek"].isin([5,6])
        ).astype(int)



        # Arrange columns exactly like training

        df = df[
        [
        'Product',
        'Quantity',
        'UnitPrice',
        'ShippingAddress',
        'PaymentMethod',
        'OrderStatus',
        'TrackingNumber',
        'ItemsInCart',
        'CouponCode',
        'ReferralSource',
        'TotalPrice',
        'Year',
        'Month',
        'Day',
        'DayOfWeek',
        'PricePerItem',
        'LargeCart',
        'HighValueOrder',
        'CouponUsed',
        'WeekendOrder'
        ]
        ]


        prediction = model.predict(df)[0]


        probability = model.predict_proba(df)[0][1]


        result = "FRAUD" if prediction == 1 else "LEGITIMATE"


        return jsonify({

            "Prediction": result,

            "Fraud Probability": round(
                float(probability),4
            )

        })


    except Exception as e:

        return jsonify({

            "error":str(e)

        })


if __name__=="__main__":

    app.run(
        debug=True,
        port=5000
    )