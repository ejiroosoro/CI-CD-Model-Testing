"""
train.py
--------
Trains a simple Iris flower classifier using scikit-learn.
Saves the trained model as model.pkl for testing and reuse.

Usage:
    python train.py
"""

import pickle
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = Path("model.pkl")


def train() -> dict:
    """
    Loads the Iris dataset, trains a Random Forest classifier,
    saves model.pkl, and returns training results.

    Returns:
        dict with keys: accuracy, model_path, n_samples, n_features, classes
    """
    print("[train] Loading Iris dataset ...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"[train] Dataset — {len(X)} samples, {X.shape[1]} features, {len(iris.target_names)} classes")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("[train] Training Random Forest classifier ...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[train] Accuracy: {accuracy * 100:.2f}%")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[train] Model saved → {MODEL_PATH}")

    return {
        "accuracy":   accuracy,
        "model_path": str(MODEL_PATH),
        "n_samples":  len(X),
        "n_features": X.shape[1],
        "classes":    list(iris.target_names),
    }


if __name__ == "__main__":
    results = train()
    print("\n--- Training Summary ---")
    for key, value in results.items():
        print(f"  {key}: {value}")
