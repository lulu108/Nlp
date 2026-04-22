from __future__ import annotations

import joblib
import pandas as pd
import pytest

from algorithms import classifier
from algorithms.classifier_preprocessing import build_classifier_text, preprocess_classifier_text
from scripts.train_classifier import load_training_dataframe


def test_load_classifier_artifacts_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier, "VECTORIZER_PATH", tmp_path / "tfidf_vectorizer.pkl")
    monkeypatch.setattr(classifier, "MODEL_PATH", tmp_path / "svm_model.pkl")
    monkeypatch.setattr(classifier, "_ARTIFACT_CACHE", None)

    with pytest.raises(FileNotFoundError):
        classifier.load_classifier_artifacts(force_reload=True)


def test_preprocess_classifier_text_segments_chinese_text():
    processed = preprocess_classifier_text("\u82af\u7247\u6280\u672f\u521b\u65b0\u6b63\u5728\u52a0\u901f")

    assert processed
    assert " " in processed
    assert "\u82af\u7247" in processed
    assert "\u6280\u672f" in processed


def test_load_training_dataframe_combines_title_and_applies_preprocessing(tmp_path):
    data_path = tmp_path / "classify_train.csv"
    pd.DataFrame(
        {
            "title": [
                "\u82af\u7247\u53d1\u5e03\u4f1a",
                "\u79d1\u6280\u5cf0\u4f1a",
                "\u7403\u961f\u8bad\u7ec3",
                "\u8054\u8d5b\u524d\u77bb",
            ],
            "text": [
                "\u4eba\u5de5\u667a\u80fd\u82af\u7247\u6280\u672f\u518d\u6b21\u5347\u7ea7",
                "\u7b97\u6cd5\u6a21\u578b\u5728\u65b0\u5e73\u53f0\u5b8c\u6210\u90e8\u7f72",
                "\u8054\u8d5b\u51b3\u8d5b\u524d\u7403\u961f\u52a0\u7d27\u5907\u6218",
                "\u524d\u950b\u7403\u5458\u5728\u70ed\u8eab\u8d5b\u4e2d\u72b6\u6001\u706b\u70ed",
            ],
            "label": ["\u79d1\u6280", "\u79d1\u6280", "\u4f53\u80b2", "\u4f53\u80b2"],
        }
    ).to_csv(data_path, index=False, encoding="utf-8-sig")

    out = load_training_dataframe(data_path)

    assert out.loc[0, "text"] == build_classifier_text(
        "\u82af\u7247\u53d1\u5e03\u4f1a",
        "\u4eba\u5de5\u667a\u80fd\u82af\u7247\u6280\u672f\u518d\u6b21\u5347\u7ea7",
    )
    assert out.loc[2, "text"] == build_classifier_text(
        "\u7403\u961f\u8bad\u7ec3",
        "\u8054\u8d5b\u51b3\u8d5b\u524d\u7403\u961f\u52a0\u7d27\u5907\u6218",
    )
    assert "\u82af\u7247" in out.loc[0, "text"]
    assert " " in out.loc[0, "text"]


def test_classify_text_returns_label_and_confidence(tmp_path, monkeypatch):
    class FakeVectorizer:
        def __init__(self) -> None:
            self.last_input: list[str] | None = None

        def transform(self, texts):
            self.last_input = list(texts)
            return [[1.0]]

    class FakeModel:
        def predict(self, features):
            return ["\u79d1\u6280"]

        def predict_proba(self, features):
            return [[0.9, 0.1]]

    vectorizer = FakeVectorizer()
    model = FakeModel()
    monkeypatch.setattr(classifier, "load_classifier_artifacts", lambda force_reload=False: (vectorizer, model))

    raw_text = "\u82af\u7247\u6280\u672f\u521b\u65b0\u6b63\u5728\u52a0\u901f"
    label, confidence = classifier.classify_text(raw_text)

    assert isinstance(label, str)
    assert label == "\u79d1\u6280"
    assert confidence == 0.9
    assert vectorizer.last_input == [preprocess_classifier_text(raw_text)]


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
