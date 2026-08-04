import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt


# Page Configuration

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)


# Load Model

model = joblib.load(
    "models/best_gradient_boosting.pkl"
)


# Title

st.title(" Fraud Detection System")

st.write(
    "Machine Learning Based Fraud Transaction Detection Pipeline"
)


st.divider()


# Model Performance


st.subheader("Model Performance")


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Accuracy",
    "94.58%"
)


col2.metric(
    "ROC-AUC",
    "96.86%"
)


col3.metric(
    "Recall",
    "83.33%"
)


col4.metric(
    "F1 Score",
    "60.61%"
)



st.divider()



# Feature Importance


st.subheader(
    "Feature Importance"
)


feature_path = (
"output/charts/feature_importance.png"
)


if os.path.exists(feature_path):

    st.image(
        feature_path,
        width=800
    )

else:

    st.warning(
        "Feature importance chart not found"
    )



st.divider()



# Transaction Prediction


st.subheader(
    " Single Transaction Prediction"
)



col1,col2,col3 = st.columns(3)



with col1:

    product = st.number_input(
        "Product",
        value=2
    )


    quantity = st.number_input(
        "Quantity",
        value=3
    )


    unit_price = st.number_input(
        "Unit Price",
        value=40
    )


    shipping_address = st.number_input(
        "Shipping Address",
        value=8
    )


    payment_method = st.number_input(
        "Payment Method",
        value=1
    )



with col2:

    order_status = st.number_input(
        "Order Status",
        value=0
    )


    tracking_number = st.number_input(
        "Tracking Number",
        value=100
    )


    items_cart = st.number_input(
        "Items In Cart",
        value=4
    )


    coupon_code = st.number_input(
        "Coupon Code",
        value=3
    )


    referral_source = st.number_input(
        "Referral Source",
        value=2
    )



with col3:

    total_price = st.number_input(
        "Total Price",
        value=1350
    )


    year = st.number_input(
        "Year",
        value=2024
    )


    month = st.number_input(
        "Month",
        value=5
    )


    day = st.number_input(
        "Day",
        value=17
    )


    day_week = st.number_input(
        "Day Of Week",
        value=5
    )




if st.button("Predict Transaction"):


    data = pd.DataFrame({

        "Product":[product],
        "Quantity":[quantity],
        "UnitPrice":[unit_price],
        "ShippingAddress":[shipping_address],
        "PaymentMethod":[payment_method],
        "OrderStatus":[order_status],
        "TrackingNumber":[tracking_number],
        "ItemsInCart":[items_cart],
        "CouponCode":[coupon_code],
        "ReferralSource":[referral_source],
        "TotalPrice":[total_price],
        "Year":[year],
        "Month":[month],
        "Day":[day],
        "DayOfWeek":[day_week]

    })



    # Feature Engineering


    data["PricePerItem"] = (
        data["TotalPrice"] /
        data["Quantity"]
    )


    data["LargeCart"] = (
        data["ItemsInCart"] > 10
    ).astype(int)


    data["HighValueOrder"] = (
        data["TotalPrice"] > 5000
    ).astype(int)


    data["CouponUsed"] = (
        data["CouponCode"] != 0
    ).astype(int)


    data["WeekendOrder"] = (
        data["DayOfWeek"].isin([5,6])
    ).astype(int)



    prediction = model.predict(data)[0]


    probability = model.predict_proba(data)[0][1]



    st.subheader(
        "Prediction Result"
    )



    if prediction == 1:

        st.error(
            "⚠ FRAUD TRANSACTION"
        )

    else:

        st.success(
            "LEGITIMATE TRANSACTION"
        )


    st.metric(
        "Fraud Probability",
        f"{probability*100:.2f}%"
    )