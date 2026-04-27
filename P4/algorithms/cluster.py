from __future__ import annotations

from typing import Iterable

import numpy as np
from algorithms.classifier_preprocessing import build_classifier_text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_CLUSTER_COUNT = 2
DEFAULT_MAX_FEATURES = 5000
DEFAULT_NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42


def _prepare_documents(documents: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    prepared_docs: list[dict[str, str]] = []

    for idx, doc in enumerate(documents):
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()

        if not title:
            raise ValueError(f"document at index {idx} is missing title")
        if not text:
            raise ValueError(f"document at index {idx} is missing text")

        processed_text = build_classifier_text(title, text)
        if not processed_text:
            raise ValueError(f"document at index {idx} is empty after preprocessing")

        prepared_docs.append(
            {
                "title": title,
                "text": text,
                "processed_text": processed_text,
            }
        )

    if len(prepared_docs) < 2:
        raise ValueError("at least two documents are required")

    return prepared_docs


def _resolve_cluster_count(cluster_count: int | None, document_count: int) -> int:
    k = DEFAULT_CLUSTER_COUNT if cluster_count is None else cluster_count

    if k < 2:
        raise ValueError("cluster_count must be >= 2")
    if k > document_count:
        raise ValueError("cluster_count must be <= document count")

    return k


def _project_points(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)

    component_count = min(2, matrix.shape[0], matrix.shape[1])
    if component_count <= 0:
        return np.zeros((matrix.shape[0], 2), dtype=float)

    reduced = PCA(n_components=component_count).fit_transform(matrix)
    if component_count == 1:
        reduced = np.column_stack([reduced[:, 0], np.zeros(reduced.shape[0], dtype=float)])

    return reduced[:, :2]


def cluster_documents(
    documents: list[dict[str, str]], cluster_count: int | None = None
) -> list[dict[str, object]]:
    """Cluster documents with TF-IDF, KMeans, and PCA for 2D visualization."""
    prepared_docs = _prepare_documents(documents)
    k = _resolve_cluster_count(cluster_count, len(prepared_docs))

    vectorizer = TfidfVectorizer(
        max_features=DEFAULT_MAX_FEATURES,
        ngram_range=DEFAULT_NGRAM_RANGE,
        token_pattern=r"(?u)\b\w+\b",
    )
    features = vectorizer.fit_transform(doc["processed_text"] for doc in prepared_docs)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(features)

    reduced_points = _project_points(features.toarray())

    points: list[dict[str, object]] = []
    for idx, doc in enumerate(prepared_docs):
        points.append(
            {
                "title": doc["title"],
                "x": round(float(reduced_points[idx, 0]), 4),
                "y": round(float(reduced_points[idx, 1]), 4),
                "cluster": int(labels[idx]),
            }
        )

    return points
