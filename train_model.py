"""
train_model.py
---------------
Generates a realistic synthetic healthcare dataset and trains a
RandomForestClassifier to predict one of four health risk categories
from three blood test features: Glucose, Haemoglobin, and Cholesterol.

This script is self-contained and can be re-run at any time to
regenerate the dataset and retrain the model:

    python train_model.py

Output artifacts:
    data/synthetic_health_data.csv   - the generated training dataset
    models/health_model.pkl          - the trained, serialized model bundle
"""

from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data")
MODELS_DIR = os.path.join(_BASE_DIR, "models")

DATA_PATH = os.path.join(DATA_DIR, "synthetic_health_data.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "health_model.pkl")

RANDOM_SEED = 42
N_SAMPLES = 1200  # at least 1000, as required

# Clinically-inspired reference ranges used purely to shape synthetic data.
# Normal ranges (approximate, for a general adult population):
#   Glucose (fasting):     70 - 99 mg/dL   normal | 100-125 prediabetes | 126+ diabetes
#   Haemoglobin:            13 - 17 g/dL (M), 12-15.5 g/dL (F) | low => anemia
#   Cholesterol (total):   <200 mg/dL desirable | 200-239 borderline | 240+ high

CLASS_LABELS = [
    "Healthy",
    "Prediabetes Risk",
    "Anemia Risk",
    "High Cholesterol Risk",
]

REMARKS_MAP = {
    "Healthy": (
        "Blood test values appear within normal ranges. "
        "Continue maintaining a healthy lifestyle."
    ),
    "Prediabetes Risk": (
        "Elevated glucose levels detected. Consider consulting a "
        "healthcare professional for further evaluation."
    ),
    "Anemia Risk": (
        "Low haemoglobin levels detected. Further medical assessment "
        "may be recommended."
    ),
    "High Cholesterol Risk": (
        "Elevated cholesterol levels detected. Lifestyle modifications "
        "and medical consultation may be beneficial."
    ),
}


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic healthcare dataset with realistic, slightly
    overlapping distributions for each of the four target classes.

    Each class is generated from class-specific Gaussian distributions
    for Glucose, Haemoglobin, and Cholesterol, then mild noise and a
    small fraction of borderline/overlapping samples are added so the
    classifier learns a realistic, non-trivial decision boundary
    rather than perfectly separable clusters.

    Args:
        n_samples: Total number of records to generate (split evenly
            across the four classes, with rounding handled gracefully).
        random_seed: Seed for NumPy's random generator, for reproducibility.

    Returns:
        A pandas DataFrame with columns:
            glucose, haemoglobin, cholesterol, label
    """
    rng = np.random.default_rng(random_seed)

    n_per_class = n_samples // len(CLASS_LABELS)
    records = []

    # --- Healthy: all three markers comfortably within normal range -------
    healthy = pd.DataFrame({
        "glucose": rng.normal(90, 8, n_per_class).clip(70, 99),
        "haemoglobin": rng.normal(14.5, 1.0, n_per_class).clip(12.5, 17.5),
        "cholesterol": rng.normal(170, 20, n_per_class).clip(120, 199),
        "label": "Healthy",
    })
    records.append(healthy)

    # --- Prediabetes Risk: elevated glucose, other markers normal-ish ------
    prediabetes = pd.DataFrame({
        "glucose": rng.normal(140, 18, n_per_class).clip(100, 250),
        "haemoglobin": rng.normal(14.0, 1.2, n_per_class).clip(11.5, 17.0),
        "cholesterol": rng.normal(185, 25, n_per_class).clip(130, 230),
        "label": "Prediabetes Risk",
    })
    records.append(prediabetes)

    # --- Anemia Risk: low haemoglobin, other markers normal-ish ------------
    anemia = pd.DataFrame({
        "glucose": rng.normal(92, 10, n_per_class).clip(70, 120),
        "haemoglobin": rng.normal(9.5, 1.5, n_per_class).clip(5.0, 11.5),
        "cholesterol": rng.normal(175, 22, n_per_class).clip(120, 220),
        "label": "Anemia Risk",
    })
    records.append(anemia)

    # --- High Cholesterol Risk: elevated cholesterol, other markers normal -
    high_chol = pd.DataFrame({
        "glucose": rng.normal(95, 10, n_per_class).clip(70, 125),
        "haemoglobin": rng.normal(14.2, 1.1, n_per_class).clip(12.0, 17.0),
        "cholesterol": rng.normal(255, 30, n_per_class).clip(230, 400),
        "label": "High Cholesterol Risk",
    })
    records.append(high_chol)

    df = pd.concat(records, ignore_index=True)

    # Inject a small proportion (~4%) of borderline/noisy samples by
    # nudging a random subset of rows with extra Gaussian jitter. This
    # mimics real-world measurement noise and avoids an unrealistically
    # perfect separation between classes.
    noise_idx = rng.choice(df.index, size=int(len(df) * 0.04), replace=False)
    df.loc[noise_idx, "glucose"] += rng.normal(0, 15, len(noise_idx))
    df.loc[noise_idx, "haemoglobin"] += rng.normal(0, 1.0, len(noise_idx))
    df.loc[noise_idx, "cholesterol"] += rng.normal(0, 20, len(noise_idx))

    # Re-clip to the overall valid form ranges after noise injection so
    # generated data always stays within plausible bounds.
    df["glucose"] = df["glucose"].clip(50, 500)
    df["haemoglobin"] = df["haemoglobin"].clip(5, 20)
    df["cholesterol"] = df["cholesterol"].clip(100, 400)

    # Round to one decimal place to resemble realistic lab report precision.
    df["glucose"] = df["glucose"].round(1)
    df["haemoglobin"] = df["haemoglobin"].round(1)
    df["cholesterol"] = df["cholesterol"].round(1)

    # Shuffle rows so the CSV isn't grouped by class in sequential blocks.
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_random_forest(
    df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train a RandomForestClassifier on the synthetic dataset.

    Args:
        df: DataFrame containing glucose, haemoglobin, cholesterol, label.
        random_seed: Seed used for the train/test split and the forest.

    Returns:
        A tuple of (trained_model, evaluation_metrics_dict).
    """
    feature_cols = ["glucose", "haemoglobin", "cholesterol"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_seed,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "feature_importances": dict(zip(feature_cols, model.feature_importances_)),
    }

    return model, metrics


def save_model_bundle(model: RandomForestClassifier, path: str = MODEL_PATH) -> None:
    """
    Persist the trained model along with metadata needed at inference
    time (feature order and class labels) using Joblib.

    Args:
        model: The trained RandomForestClassifier.
        path: Destination file path for the serialized bundle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": ["glucose", "haemoglobin", "cholesterol"],
        "class_labels": CLASS_LABELS,
        "remarks_map": REMARKS_MAP,
    }
    joblib.dump(bundle, path)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the dataset, train the model, evaluate it, and save artifacts."""
    print("Generating synthetic healthcare dataset...")
    df = generate_synthetic_dataset()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} synthetic records to: {DATA_PATH}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nTraining RandomForestClassifier...")
    model, metrics = train_random_forest(df)

    print(f"\nTest accuracy: {metrics['accuracy']:.4f}\n")
    print("Classification report:")
    print(metrics["classification_report"])

    print("Feature importances:")
    for feature, importance in metrics["feature_importances"].items():
        print(f"  {feature}: {importance:.4f}")

    save_model_bundle(model)
    print(f"\nModel bundle saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
