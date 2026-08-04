import os
import joblib
import warnings
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

warnings.filterwarnings("ignore")

# Paths

INPUT_PATH = "data/processed/feature_engineered_data.csv"

MODEL_PATH = "models/best_gradient_boosting.pkl"

REPORT_PATH = "output/reports"

os.makedirs("models", exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

# Load Dataset

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully")

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

# Apply SMOTE ONLY on Training Data

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(

    X_train,

    y_train

)

print("\nTraining Distribution")

print(pd.Series(y_train).value_counts())

# Base Model

gb = GradientBoostingClassifier(random_state=42)

# Hyperparameters

param_grid = {

    "n_estimators": [100, 150, 200],

    "learning_rate": [0.01, 0.05, 0.1],

    "max_depth": [3, 4, 5],

    "subsample": [0.8, 1.0]

}

# Grid Search

print("\nRunning GridSearchCV...")

grid = GridSearchCV(

    estimator=gb,

    param_grid=param_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1,

    verbose=1

)

grid.fit(

    X_train,

    y_train

)

print("\nBest Parameters")

print(grid.best_params_)

print("\nBest Cross Validation Score")

print(grid.best_score_)

# Best Model

best_model = grid.best_estimator_

predictions = best_model.predict(X_test)

probabilities = best_model.predict_proba(X_test)[:,1]

# Metrics

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1 = f1_score(y_test, predictions)

roc = roc_auc_score(y_test, probabilities)

print("\nMODEL PERFORMANCE")

print("="*50)

print("Accuracy :", round(accuracy,4))

print("Precision :", round(precision,4))

print("Recall :", round(recall,4))

print("F1 Score :", round(f1,4))

print("ROC-AUC :", round(roc,4))

print("\nClassification Report\n")

print(classification_report(

    y_test,

    predictions

))

# Save Model

joblib.dump(

    best_model,

    MODEL_PATH

)

print("\nBest Model Saved")

print(MODEL_PATH)

# Save Metrics

results = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC"

    ],

    "Score":[

        accuracy,

        precision,

        recall,

        f1,

        roc

    ]

})

results.to_csv(

    f"{REPORT_PATH}/tuned_model_results.csv",

    index=False

)

print("\nResults Saved")

print("output/reports/tuned_model_results.csv")