from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from algorithms.classifier_preprocessing import preprocess_classifier_text


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "classifier"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "svm_model.pkl"

_ARTIFACT_CACHE: tuple[object, object] | None = None

def load_classifier_artifacts(force_reload: bool = False) -> tuple[object, object]:
    global _ARTIFACT_CACHE

    if _ARTIFACT_CACHE is not None and not force_reload:
        return _ARTIFACT_CACHE

    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Classifier artifacts not found. "
            f"Expected files: {VECTORIZER_PATH} and {MODEL_PATH}. "
            "Please run: python scripts/train_classifier.py --data-path data/train/classify_train.csv"
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    _ARTIFACT_CACHE = (vectorizer, model)
    return _ARTIFACT_CACHE


def _confidence_from_decision(decision: np.ndarray) -> float:
    scores = np.array(decision, dtype=float)
    if scores.ndim == 0:
        return 0.5
    if scores.ndim == 1:
        if scores.shape[0] == 1:
            return 0.5
        if scores.shape[0] == 2:
            margin = abs(scores[1] - scores[0])
            return float(1.0 / (1.0 + np.exp(-margin)))

        stabilized = scores - np.max(scores)
        probs = np.exp(stabilized) / np.sum(np.exp(stabilized))
        return float(np.max(probs))

    row = scores[0]
    return _confidence_from_decision(row)


def classify_text(text: str) -> tuple[str, float]:
    value = preprocess_classifier_text(text)
    if not value:
        raise ValueError("text cannot be empty")

    vectorizer, model = load_classifier_artifacts()
    features = vectorizer.transform([value])

    predicted = model.predict(features)[0]
    label = str(predicted)

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        if probs is not None and len(probs) > 0:
            confidence = float(np.max(probs[0]))

    if confidence is None and hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        confidence = _confidence_from_decision(np.asarray(decision))

    if confidence is None:
        confidence = 0.5

    confidence = float(np.clip(confidence, 0.0, 1.0))
    return label, round(confidence, 4)
