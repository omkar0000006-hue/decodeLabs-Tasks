import os
import joblib
import warnings
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# Paths

INPUT_PATH = "data/processed/feature_engineered_data.csv"

MODEL_DIR = "models"

REPORT_DIR = "output/reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load Dataset

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully")

print("Shape :", df.shape)

# Features & Target

X = df.drop("Fraud", axis=1)

y = df["Fraud"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Shape :", X_train.shape)

print("Testing Shape :", X_test.shape)

# Apply SMOTE ONLY on Training Data

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nTraining Distribution After SMOTE")

print(pd.Series(y_train).value_counts())

# Models

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        )

}

results = []

# Training

print("\nMODEL TRAINING")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc:.4f}")

    joblib.dump(

        model,

        f"{MODEL_DIR}/{name.replace(' ','_').lower()}.pkl"

    )

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

results = results.sort_values(

    by="ROC-AUC",

    ascending=False

)

results.to_csv(

    f"{REPORT_DIR}/model_comparison.csv",

    index=False

)

print("\n")
print("="*60)

print("MODEL COMPARISON")

print("="*60)

print(results)

print("\nBest Model")

print(results.iloc[0]["Model"])

print("\nTraining Completed Successfully")