import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess

DATA_PATH = os.path.join("..", "data", "reviews.csv")
MODELS_DIR = os.path.join("..", "models")


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Preprocessing text (this may take a few seconds)...")
    df["clean_text"] = df["review_text"].apply(preprocess)
    return df


def vectorize(df):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]
    return X, y, vectorizer


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "MultinomialNB": MultinomialNB(alpha=1.0),   # Laplace smoothing
        "ComplementNB": ComplementNB(alpha=1.0),      # good for imbalance
        "LinearSVC": LinearSVC(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, acc)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

    best_name = max(results, key=lambda k: results[k][1])
    print(f"\nBest model: {best_name} (accuracy={results[best_name][1]:.4f})")
    return results[best_name][0], best_name


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()
    X, y, vectorizer = vectorize(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    joblib.dump(best_model, os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    with open(os.path.join(MODELS_DIR, "model_info.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n")

    print(f"\nSaved model + vectorizer to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
