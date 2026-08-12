import joblib
from pathlib import Path

MODEL_PATH = Path("models/customer_segmentation_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
PCA_PATH = Path("models/pca.pkl")

def check_models():
    print("=" * 60)
    print("MODEL FILES")
    print("=" * 60)

    for name, path in [
        ("Clustering Model", MODEL_PATH),
        ("Scaler", SCALER_PATH),
        ("PCA", PCA_PATH),
    ]:
        status = "Available" if path.exists() else "Missing"
        print(f"\n{name} : {status}")
        print(path)

if __name__ == "__main__":
    check_models()
