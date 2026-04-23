from __future__ import annotations
# python scripts/evaluate_cluster.py --data-path data/train/classify_train.csv
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.classifier_preprocessing import build_classifier_text
from algorithms.cluster import DEFAULT_MAX_FEATURES, DEFAULT_NGRAM_RANGE, RANDOM_STATE

DEFAULT_DATA_PATH = ROOT_DIR / "data" / "train" / "classify_train.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P4 text clustering on labeled CSV data")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="CSV path with at least text,label columns",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=None,
        help="Override the cluster count. Defaults to the number of unique labels.",
    )
    return parser.parse_args()


def load_csv_auto(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as err:  # noqa: BLE001
            last_err = err
    raise ValueError(f"failed to read CSV: {path}, last error: {last_err}")


def load_evaluation_dataframe(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"evaluation data not found: {data_path}")

    df = load_csv_auto(data_path)
    if "text" not in df.columns:
        raise ValueError("evaluation data must include 'text' column")
    if "label" not in df.columns:
        raise ValueError("evaluation data must include 'label' column")

    text_series = df["text"].fillna("").astype(str)
    if "title" in df.columns:
        title_series = df["title"].fillna("").astype(str)
        processed_texts = [build_classifier_text(title, text) for title, text in zip(title_series, text_series)]
    else:
        title_series = pd.Series([""] * len(df))
        processed_texts = text_series.map(build_classifier_text)

    out = pd.DataFrame(
        {
            "title": title_series.astype(str).str.strip(),
            "text": text_series.astype(str).str.strip(),
            "label": df["label"].astype(str).str.strip(),
            "processed_text": processed_texts,
        }
    )

    out = out[(out["label"] != "") & (out["processed_text"] != "")].reset_index(drop=True)
    if out.empty:
        raise ValueError("evaluation data is empty after cleaning")
    if len(out) < 2:
        raise ValueError("at least two valid documents are required for cluster evaluation")

    label_count = int(out["label"].nunique())
    if label_count < 2:
        raise ValueError("at least two unique labels are required for cluster evaluation")

    return out


def resolve_cluster_count(dataframe: pd.DataFrame, cluster_count: int | None) -> int:
    resolved = int(dataframe["label"].nunique()) if cluster_count is None else int(cluster_count)
    document_count = len(dataframe)

    if resolved < 2:
        raise ValueError("cluster_count must be >= 2")
    if resolved > document_count:
        raise ValueError("cluster_count must be <= document count")

    return resolved


def calculate_purity(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    contingency = pd.crosstab(
        pd.Series(predicted_labels, name="cluster"),
        pd.Series(true_labels, name="label"),
    )
    return float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())


def summarize_clusters(true_labels: np.ndarray, predicted_labels: np.ndarray) -> list[dict[str, object]]:
    contingency = pd.crosstab(
        pd.Series(predicted_labels, name="cluster"),
        pd.Series(true_labels, name="label"),
    )
    summary: list[dict[str, object]] = []

    for cluster_id, row in contingency.iterrows():
        cluster_size = int(row.sum())
        dominant_label = str(row.idxmax())
        dominant_count = int(row.max())
        summary.append(
            {
                "cluster": int(cluster_id),
                "size": cluster_size,
                "dominant_label": dominant_label,
                "dominant_count": dominant_count,
                "dominant_ratio": dominant_count / cluster_size if cluster_size > 0 else 0.0,
            }
        )

    return summary


def evaluate_clustering(dataframe: pd.DataFrame, cluster_count: int) -> dict[str, object]:
    vectorizer = TfidfVectorizer(
        max_features=DEFAULT_MAX_FEATURES,
        ngram_range=DEFAULT_NGRAM_RANGE,
        token_pattern=r"(?u)\b\w+\b",
    )
    features = vectorizer.fit_transform(dataframe["processed_text"])

    model = KMeans(n_clusters=cluster_count, n_init=10, random_state=RANDOM_STATE)
    predicted_labels = model.fit_predict(features)
    true_labels = dataframe["label"].to_numpy()

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    unique_clusters = np.unique(predicted_labels)
    if 1 < len(unique_clusters) < len(dataframe):
        silhouette = float(silhouette_score(features, predicted_labels))
    else:
        silhouette = float("nan")

    purity = calculate_purity(true_labels, predicted_labels)
    cluster_summary = summarize_clusters(true_labels, predicted_labels)

    return {
        "n_documents": int(len(dataframe)),
        "n_labels": int(dataframe["label"].nunique()),
        "cluster_count": int(cluster_count),
        "ari": float(ari),
        "nmi": float(nmi),
        "silhouette": silhouette,
        "purity": float(purity),
        "cluster_summary": cluster_summary,
    }


def print_summary(metrics: dict[str, object]) -> None:
    print("===== Cluster Evaluation Summary =====")
    print(f"Documents: {metrics['n_documents']}")
    print(f"Unique labels: {metrics['n_labels']}")
    print(f"Cluster count: {metrics['cluster_count']}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")

    silhouette = float(metrics["silhouette"])
    if np.isnan(silhouette):
        print("Silhouette Score: nan")
    else:
        print(f"Silhouette Score: {silhouette:.4f}")

    print(f"Purity: {metrics['purity']:.4f}")
    print("Per-cluster dominant labels:")

    for item in metrics["cluster_summary"]:
        print(
            "  "
            f"cluster={item['cluster']} | "
            f"size={item['size']} | "
            f"dominant_label={item['dominant_label']} | "
            f"dominant_count={item['dominant_count']} | "
            f"dominant_ratio={item['dominant_ratio']:.4f}"
        )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    dataframe = load_evaluation_dataframe(data_path)
    cluster_count = resolve_cluster_count(dataframe, args.cluster_count)
    metrics = evaluate_clustering(dataframe, cluster_count)
    print_summary(metrics)


if __name__ == "__main__":
    main()
