"""
model.py
--------
Inference layer for the Health Prediction Application.

Loads the trained RandomForestClassifier bundle (produced by
train_model.py) and exposes a simple `predict_health_risk` function
that the Streamlit UI calls whenever a patient record is created or
updated. If the model artifact is missing, this module will
automatically trigger training so the app remains usable out of the box.
"""

from __future__ import annotations

import os
from typing import Tuple

import joblib
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "health_model.pkl")

# Module-level cache so the (relatively expensive) joblib.load only
# happens once per process, even though Streamlit reruns the script
# top-to-bottom on every interaction.
_model_bundle: dict | None = None


def _load_bundle() -> dict:
    """
    Load the serialized model bundle from disk, training it on first
    use if the artifact does not yet exist.

    Returns:
        A dictionary containing the model, feature column order,
        class labels, and the remarks lookup map.
    """
    global _model_bundle

    if _model_bundle is not None:
        return _model_bundle

    if not os.path.exists(MODEL_PATH):
        # Lazily train the model if it hasn't been generated yet, so a
        # fresh checkout of the project works without a manual step.
        from train_model import generate_synthetic_dataset, train_random_forest, save_model_bundle

        df = generate_synthetic_dataset()
        trained_model, _ = train_random_forest(df)
        save_model_bundle(trained_model)

    _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def predict_health_risk(
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
) -> Tuple[str, str]:
    """
    Predict the health risk category for a single patient's blood
    test values and return the corresponding remark text.

    Args:
        glucose: Glucose level in mg/dL.
        haemoglobin: Haemoglobin level in g/dL.
        cholesterol: Cholesterol level in mg/dL.

    Returns:
        A tuple of (prediction_label, remarks_text), e.g.
        ("Prediabetes Risk", "Elevated glucose levels detected. ...").
    """
    bundle = _load_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    remarks_map = bundle["remarks_map"]

    # Build the input as a single-row DataFrame with the exact column
    # order/names the model was trained on, to avoid sklearn feature
    # name mismatch warnings.
    input_df = pd.DataFrame(
        [[glucose, haemoglobin, cholesterol]],
        columns=feature_columns,
    )

    prediction = model.predict(input_df)[0]
    remarks = remarks_map.get(
        prediction,
        "Unable to generate a specific remark for this prediction.",
    )

    return prediction, remarks


def predict_with_probabilities(
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
) -> Tuple[str, str, dict]:
    """
    Same as predict_health_risk, but also returns the model's class
    probability distribution -- useful for displaying prediction
    confidence in the UI.

    Args:
        glucose: Glucose level in mg/dL.
        haemoglobin: Haemoglobin level in g/dL.
        cholesterol: Cholesterol level in mg/dL.

    Returns:
        A tuple of (prediction_label, remarks_text, probabilities_dict)
        where probabilities_dict maps each class label to its predicted
        probability (0.0 - 1.0).
    """
    bundle = _load_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    remarks_map = bundle["remarks_map"]
    class_labels = bundle["class_labels"]

    input_df = pd.DataFrame(
        [[glucose, haemoglobin, cholesterol]],
        columns=feature_columns,
    )

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    # model.classes_ holds the order corresponding to predict_proba's columns.
    probabilities = {label: float(p) for label, p in zip(model.classes_, proba)}
    # Ensure every known class label is represented, even with 0.0 probability.
    for label in class_labels:
        probabilities.setdefault(label, 0.0)

    remarks = remarks_map.get(
        prediction,
        "Unable to generate a specific remark for this prediction.",
    )

    return prediction, remarks, probabilities
