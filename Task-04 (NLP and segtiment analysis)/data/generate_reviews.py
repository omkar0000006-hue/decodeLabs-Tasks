import pandas as pd
import random

random.seed(42)

INPUT_PATH = "orders_dataset.csv"
OUTPUT_PATH = "reviews.csv"

POSITIVE_TEMPLATES = [
    "The {product} exceeded my expectations, great quality and fast shipping!",
    "Absolutely love this {product}, works perfectly and arrived on time.",
    "Excellent {product}, would definitely buy again. Highly recommend!",
    "This {product} is not bad at all, actually pretty amazing for the price.",
    "Super happy with my {product} purchase, no complaints so far.",
    "Great value for money, the {product} works flawlessly.",
    "I was worried at first but the {product} turned out to be really good.",
    "Fast delivery and the {product} feels premium, very satisfied.",
    "Best {product} I have bought this year, five stars.",
    "The {product} is not expensive but it does not feel cheap either. Loved it.",
]

NEGATIVE_TEMPLATES = [
    "The {product} was not good, arrived damaged and support was unhelpful.",
    "Terrible experience, the {product} stopped working within a week.",
    "I am not happy with this {product}, complete waste of money.",
    "The {product} is not what I expected, poor quality overall.",
    "Disappointed with the {product}, had to return it immediately.",
    "Awful build quality, would not recommend this {product} to anyone.",
    "The {product} never worked properly from day one.",
    "Shipping took forever and the {product} was not even functional.",
    "This {product} is not worth the price, very disappointing.",
    "Regret buying this {product}, customer service was no help either.",
]

NEUTRAL_LEANING_POSITIVE = [
    "The {product} is okay, does the job, nothing special but not bad either.",
]
NEUTRAL_LEANING_NEGATIVE = [
    "The {product} is still stuck in processing, not sure if it is worth the wait.",
]

STATUS_SENTIMENT_BIAS = {
    "Delivered": 0.85,   # 85% chance positive
    "Shipped": 0.75,
    "Pending": 0.5,
    "Returned": 0.10,
    "Cancelled": 0.05,
}


def make_review(product, status):
    bias = STATUS_SENTIMENT_BIAS.get(status, 0.5)
    is_positive = random.random() < bias

    if status == "Pending":
        template = random.choice(
            POSITIVE_TEMPLATES if is_positive else NEGATIVE_TEMPLATES
        )
    else:
        template = random.choice(
            POSITIVE_TEMPLATES if is_positive else NEGATIVE_TEMPLATES
        )

    text = template.format(product=product.lower())
    sentiment = "Positive" if is_positive else "Negative"
    return text, sentiment


def main():
    df = pd.read_csv(INPUT_PATH)

    reviews = []
    for i, row in df.iterrows():
        text, sentiment = make_review(row["Product"], row["OrderStatus"])
        reviews.append(
            {
                "review_id": row["OrderID"],
                "product": row["Product"],
                "order_status": row["OrderStatus"],
                "review_text": text,
                "sentiment": sentiment,
            }
        )

    out_df = pd.DataFrame(reviews)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(out_df)} reviews -> {OUTPUT_PATH}")
    print(out_df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
