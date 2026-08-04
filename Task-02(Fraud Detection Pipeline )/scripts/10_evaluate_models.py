import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Paths

DATA_PATH = "data/processed/feature_engineered_data.csv"

MODEL_DIR = "models"

REPORT_DIR = "output/reports"

CHART_DIR = "output/charts"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# Load Dataset

df = pd.read_csv(DATA_PATH)

X = df.drop("Fraud", axis=1)

y = df["Fraud"]

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Models

models = {

    "Logistic Regression":"logistic_regression.pkl",

    "Decision Tree":"decision_tree.pkl",

    "Random Forest":"random_forest.pkl",

    "Gradient Boosting":"gradient_boosting.pkl",

    "KNN":"knn.pkl",

    "SVM":"svm.pkl"

}

results = []

# Evaluate

for name,file in models.items():

    print("\n"+"="*60)

    print(name)

    print("="*60)

    model = joblib.load(f"{MODEL_DIR}/{file}")

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test,pred)

    precision = precision_score(y_test,pred)

    recall = recall_score(y_test,pred)

    f1 = f1_score(y_test,pred)

    roc = roc_auc_score(y_test,prob)

    print("Accuracy :",round(accuracy,4))

    print("Precision :",round(precision,4))

    print("Recall :",round(recall,4))

    print("F1 :",round(f1,4))

    print("ROC AUC :",round(roc,4))

    print("\nClassification Report\n")

    print(classification_report(y_test,pred))

    # Confusion Matrix

    cm = confusion_matrix(y_test,pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title(name)

    plt.savefig(
        f"{CHART_DIR}/{file.replace('.pkl','')}_cm.png"
    )

    plt.close()

    # ROC Curve

    RocCurveDisplay.from_predictions(

        y_test,

        prob

    )

    plt.title(name)

    plt.savefig(

        f"{CHART_DIR}/{file.replace('.pkl','')}_roc.png"

    )

    plt.close()

    results.append({

        "Model":name,

        "Accuracy":accuracy,

        "Precision":precision,

        "Recall":recall,

        "F1 Score":f1,

        "ROC-AUC":roc

    })

# Save Results

results = pd.DataFrame(results)

results.to_csv(

    f"{REPORT_DIR}/evaluation_results.csv",

    index=False

)

print("\nEvaluation Completed Successfully")

print("\nResults Saved")

print("output/reports/evaluation_results.csv")

print("\nCharts Saved")

print("output/charts/")