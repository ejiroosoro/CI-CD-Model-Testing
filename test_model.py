"""
test_model.py
-------------
Pytest tests for the trained Iris classifier.
Tests run automatically via GitHub Actions on every push.

Usage:
    pytest test_model.py -v
"""

import pickle
import pytest
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from train import train

MODEL_PATH = Path("model.pkl")


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """
    Trains the model once per test session and saves model.pkl.
    All tests share this fixture.
    """
    train()
    yield
    # Cleanup after all tests finish
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()


@pytest.fixture(scope="session")
def model(trained_model):
    """Loads and returns the saved model."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def test_data():
    """Returns the Iris test split used for evaluation."""
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, y_test


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

def test_model_file_exists(trained_model):
    """model.pkl must exist after training."""
    assert MODEL_PATH.exists(), "model.pkl was not created by train.py"


def test_model_loads(model):
    """model.pkl must be loadable as a valid sklearn model."""
    assert model is not None
    assert hasattr(model, "predict"), "Loaded object is not a valid sklearn model"


def test_model_accuracy_above_90(model, test_data):
    """Model accuracy must be greater than 90%."""
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n  Model accuracy: {accuracy * 100:.2f}%")
    assert accuracy > 0.90, (
        f"Accuracy {accuracy * 100:.2f}% is below the required 90% threshold."
    )


def test_model_predicts_correct_classes(model):
    """Model must only predict valid Iris class labels (0, 1, 2)."""
    iris = load_iris()
    predictions = model.predict(iris.data)
    valid_classes = set(range(len(iris.target_names)))
    predicted_classes = set(predictions)
    assert predicted_classes.issubset(valid_classes), (
        f"Model predicted invalid classes: {predicted_classes - valid_classes}"
    )


def test_model_prediction_shape(model, test_data):
    """Prediction output shape must match number of test samples."""
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    assert predictions.shape == y_test.shape, (
        f"Prediction shape {predictions.shape} doesn't match expected {y_test.shape}"
    )


def test_model_single_prediction(model):
    """Model must handle a single sample input without errors."""
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Known Iris setosa
    prediction = model.predict(sample)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]


def test_model_probability_output(model):
    """Model must return valid probabilities that sum to ~1.0."""
    iris = load_iris()
    sample = iris.data[:1]
    proba = model.predict_proba(sample)
    assert proba.shape == (1, 3), "Expected 3 class probabilities"
    assert abs(proba.sum() - 1.0) < 1e-6, "Probabilities must sum to 1.0"
