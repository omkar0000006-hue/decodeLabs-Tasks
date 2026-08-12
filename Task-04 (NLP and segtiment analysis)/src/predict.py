import os
import sys
import joblib

from preprocessing import preprocess

MODELS_DIR = os.path.join("..", "models")


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    return model, vectorizer


def predict_sentiment(text: str, model, vectorizer) -> str:
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]


def main():
    model, vectorizer = load_artifacts()

    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    else:
        texts = [
            "This product is amazing, I love it so much!",
            "This product is not good, it stopped working immediately.",
            "Terrible quality, would not recommend to anyone.",
            "Not bad at all, actually really impressed with the build.",
        ]

    for t in texts:
        pred = predict_sentiment(t, model, vectorizer)
        print(f"Review : {t}")
        print(f"Sentiment: {pred}\n")


if __name__ == "__main__":
    main()
