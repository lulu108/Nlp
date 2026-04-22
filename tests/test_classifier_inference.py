from __future__ import annotations

import joblib
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from algorithms import classifier


def test_load_classifier_artifacts_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier, "VECTORIZER_PATH", tmp_path / "tfidf_vectorizer.pkl")
    monkeypatch.setattr(classifier, "MODEL_PATH", tmp_path / "svm_model.pkl")
    monkeypatch.setattr(classifier, "_ARTIFACT_CACHE", None)

    with pytest.raises(FileNotFoundError):
        classifier.load_classifier_artifacts(force_reload=True)


def test_classify_text_returns_label_and_confidence(tmp_path, monkeypatch):
    texts = ["芯片和算法发展", "人工智能技术升级", "球队获得冠军", "联赛比赛很激烈"]
    labels = ["科技", "科技", "体育", "体育"]

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(texts)

    model = CalibratedClassifierCV(LinearSVC(max_iter=5000), method="sigmoid", cv=2)
    model.fit(x, labels)

    vec_path = tmp_path / "tfidf_vectorizer.pkl"
    model_path = tmp_path / "svm_model.pkl"
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model, model_path)

    monkeypatch.setattr(classifier, "VECTORIZER_PATH", vec_path)
    monkeypatch.setattr(classifier, "MODEL_PATH", model_path)
    monkeypatch.setattr(classifier, "_ARTIFACT_CACHE", None)

    label, confidence = classifier.classify_text("芯片技术创新正在加速")

    assert isinstance(label, str)
    assert label in {"科技", "体育"}
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_classify_text_empty_raises_value_error():
    with pytest.raises(ValueError):
        classifier.classify_text("   \n\t  ")


def test_load_classifier_artifacts_force_reload(tmp_path, monkeypatch):
    vec_path = tmp_path / "tfidf_vectorizer.pkl"
    model_path = tmp_path / "svm_model.pkl"

    persisted_vectorizer = {"name": "persisted_vectorizer"}
    persisted_model = {"name": "persisted_model"}
    joblib.dump(persisted_vectorizer, vec_path)
    joblib.dump(persisted_model, model_path)

    cached_artifacts = ("cached_vectorizer", "cached_model")

    monkeypatch.setattr(classifier, "VECTORIZER_PATH", vec_path)
    monkeypatch.setattr(classifier, "MODEL_PATH", model_path)
    monkeypatch.setattr(classifier, "_ARTIFACT_CACHE", cached_artifacts)

    no_reload = classifier.load_classifier_artifacts(force_reload=False)
    assert no_reload == cached_artifacts

    reloaded = classifier.load_classifier_artifacts(force_reload=True)
    assert reloaded != cached_artifacts
    assert reloaded[0] == persisted_vectorizer
    assert reloaded[1] == persisted_model
