import joblib
import pandas as pd

# Load Model

MODEL_PATH = "models/gradient_boosting.pkl"

model = joblib.load(MODEL_PATH)

print("\nGradient Boosting Model Loaded Successfully")

# User Input

print("\nENTER TRANSACTION DETAILS\n")

product = int(input("Product (Encoded): "))
quantity = int(input("Quantity: "))
unit_price = float(input("Unit Price: "))
shipping = int(input("Shipping Address (Encoded): "))
payment = int(input("Payment Method (Encoded): "))
order_status = int(input("Order Status (Encoded): "))
tracking = int(input("Tracking Number (Encoded): "))
items = int(input("Items In Cart: "))
coupon = int(input("Coupon Code (Encoded): "))
referral = int(input("Referral Source (Encoded): "))
total = float(input("Total Price: "))
year = int(input("Year: "))
month = int(input("Month: "))
day = int(input("Day: "))
dayofweek = int(input("Day Of Week (0-6): "))

# Engineered Features

price_per_item = total / quantity if quantity != 0 else 0

large_cart = 1 if items >= 5 else 0

high_value = 1 if total >= 1500 else 0

coupon_used = 0 if coupon == 0 else 1

weekend = 1 if dayofweek in [5, 6] else 0

# DataFrame

sample = pd.DataFrame([{

    "Product": product,

    "Quantity": quantity,

    "UnitPrice": unit_price,

    "ShippingAddress": shipping,

    "PaymentMethod": payment,

    "OrderStatus": order_status,

    "TrackingNumber": tracking,

    "ItemsInCart": items,

    "CouponCode": coupon,

    "ReferralSource": referral,

    "TotalPrice": total,

    "Year": year,

    "Month": month,

    "Day": day,

    "DayOfWeek": dayofweek,

    "PricePerItem": price_per_item,

    "LargeCart": large_cart,

    "HighValueOrder": high_value,

    "CouponUsed": coupon_used,

    "WeekendOrder": weekend

}])

# Prediction

prediction = model.predict(sample)[0]

probability = model.predict_proba(sample)[0][1]

# Output

print("\n" + "="*50)

print("FRAUD DETECTION RESULT")

print("="*50)

if prediction == 1:

    print("\nPrediction : FRAUD TRANSACTION")

else:

    print("\nPrediction : LEGITIMATE TRANSACTION")

print(f"\nFraud Probability : {probability:.2%}")

print("\nPrediction Completed Successfully")