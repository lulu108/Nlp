from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from algorithms import classifier
from algorithms.classifier_preprocessing import preprocess_classifier_text
from backend.app import create_app


def test_classify_api_integration_loads_real_artifacts(tmp_path, monkeypatch):
    texts = [
        "\u82af\u7247\u6280\u672f\u548c\u7b97\u6cd5\u6a21\u578b\u6301\u7eed\u5347\u7ea7",
        "\u4eba\u5de5\u667a\u80fd\u5e73\u53f0\u6b63\u5728\u63a8\u52a8\u79d1\u6280\u521b\u65b0",
        "\u7403\u961f\u8d62\u5f97\u8054\u8d5b\u51a0\u519b",
        "\u7403\u5458\u6b63\u5728\u79ef\u6781\u5907\u6218\u6bd4\u8d5b",
    ]
    labels = ["\u79d1\u6280", "\u79d1\u6280", "\u4f53\u80b2", "\u4f53\u80b2"]

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    features = vectorizer.fit_transform([preprocess_classifier_text(text) for text in texts])
    model = LinearSVC(max_iter=5000)
    model.fit(features, labels)

    vec_path = tmp_path / "tfidf_vectorizer.pkl"
    model_path = tmp_path / "svm_model.pkl"
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model, model_path)

    monkeypatch.setattr(classifier, "VECTORIZER_PATH", vec_path)
    monkeypatch.setattr(classifier, "MODEL_PATH", model_path)
    monkeypatch.setattr(classifier, "_ARTIFACT_CACHE", None)

    app = create_app()
    client = app.test_client()

    resp = client.post("/api/classify", json={"text": "\u4eba\u5de5\u667a\u80fd\u82af\u7247\u6280\u672f\u53d1\u5c55\u8fc5\u901f"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "label" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
