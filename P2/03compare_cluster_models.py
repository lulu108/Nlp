# -*- coding: utf-8 -*-
"""
Compare clustering models for P2:
KMeans vs MiniBatchKMeans vs Agglomerative Clustering.

Main output:
    P2/data/cluster_model_comparison_{tag}.csv
    P2/data/cluster_result_compare_{tag}.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_INPUT = DATA_DIR / "news_for_cluster_800.csv"


def find_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in columns:
        c_lower = c.lower()
        for cand in candidates:
            if cand.lower() in c_lower:
                return c
    return None


def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering ACC by Hungarian matching.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    n = max(len(labels_true), len(labels_pred))

    true_to_idx = {label: i for i, label in enumerate(labels_true)}
    pred_to_idx = {label: i for i, label in enumerate(labels_pred)}

    cost_matrix = np.zeros((n, n), dtype=np.int64)

    for yt, yp in zip(y_true, y_pred):
        i = pred_to_idx[yp]
        j = true_to_idx[yt]
        cost_matrix[i, j] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    matched = cost_matrix[row_ind, col_ind].sum()
    acc = matched / len(y_true)

    mapping = {}
    idx_to_true = {i: label for label, i in true_to_idx.items()}
    idx_to_pred = {i: label for label, i in pred_to_idx.items()}

    for r, c in zip(row_ind, col_ind):
        if r in idx_to_pred and c in idx_to_true:
            mapping[idx_to_pred[r]] = idx_to_true[c]

    return float(acc), mapping


def build_tfidf_vectorizer(
    feature_mode: str,
    max_df: float,
    min_df: int,
    ngram_min: int,
    ngram_max: int,
    max_features: int | None,
    sublinear_tf: bool,
):
    kwargs = {
        "analyzer": feature_mode,
        "max_df": max_df,
        "min_df": min_df,
        "ngram_range": (ngram_min, ngram_max),
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
    }

    if feature_mode == "word":
        kwargs["token_pattern"] = r"(?u)\b\w+\b"

    return TfidfVectorizer(**kwargs)


def evaluate_result(
    model_name: str,
    y_true,
    y_pred,
    X_for_silhouette,
    extra_info: dict | None = None,
):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc, mapping = clustering_accuracy(y_true, y_pred)

    # Silhouette requires at least 2 clusters and fewer clusters than samples.
    unique_pred = np.unique(y_pred)
    if 1 < len(unique_pred) < len(y_pred):
        sil = silhouette_score(X_for_silhouette, y_pred, metric="cosine")
    else:
        sil = np.nan

    row = {
        "model": model_name,
        "n_clusters_pred": len(unique_pred),
        "acc": round(float(acc), 6),
        "nmi": round(float(nmi), 6),
        "ari": round(float(ari), 6),
        "silhouette_cosine": round(float(sil), 6) if not np.isnan(sil) else np.nan,
        "cluster_to_label_mapping": str(mapping),
    }

    if extra_info:
        row.update(extra_info)

    return row, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--tag", default="800")
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=4)

    # Default follows your current 800 preset.
    parser.add_argument("--feature-mode", choices=["word", "char"], default="char")
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--sublinear-tf", default="true")

    parser.add_argument("--kmeans-n-init", type=int, default=120)

    parser.add_argument("--mbk-n-init", type=int, default=100)
    parser.add_argument("--mbk-batch-size", type=int, default=128)
    parser.add_argument("--mbk-max-iter", type=int, default=500)

    parser.add_argument("--svd-dim", type=int, default=100)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sublinear_tf = str(args.sublinear_tf).lower() in {"1", "true", "yes", "y", "on"}
    max_features = None if args.max_features <= 0 else args.max_features

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    label_col = find_column(df.columns, ["label", "category", "class", "tag", "type"])
    if args.feature_mode == "char":
        text_col = find_column(df.columns, ["text", "content", "text_cut"])
    else:
        text_col = find_column(df.columns, ["text_cut", "text", "content"])

    if label_col is None:
        raise ValueError("Cannot find label column.")
    if text_col is None:
        raise ValueError("Cannot find text column.")

    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.strip() != ""].copy()

    label_names = sorted(df[label_col].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(label_names)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    y_true = df[label_col].map(label_to_id).values

    vectorizer = build_tfidf_vectorizer(
        feature_mode=args.feature_mode,
        max_df=args.max_df,
        min_df=args.min_df,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )

    X = vectorizer.fit_transform(df[text_col])

    rows = []
    result_df = df[[label_col, text_col]].copy()

    common_info = {
        "tag": args.tag,
        "n_samples": X.shape[0],
        "feature_mode": args.feature_mode,
        "vocab_size": X.shape[1],
        "max_df": args.max_df,
        "min_df": args.min_df,
        "ngram_range": f"({args.ngram_min}, {args.ngram_max})",
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
    }

    # 1. KMeans
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        n_init=args.kmeans_n_init,
    )
    y_kmeans = kmeans.fit_predict(X)
    row, mapping = evaluate_result(
        model_name="KMeans",
        y_true=y_true,
        y_pred=y_kmeans,
        X_for_silhouette=X,
        extra_info={
            **common_info,
            "model_params": f"n_init={args.kmeans_n_init}",
            "representation": "TF-IDF",
        },
    )
    rows.append(row)
    result_df["cluster_kmeans"] = y_kmeans
    result_df["mapped_kmeans"] = pd.Series(y_kmeans).map(mapping).map(id_to_label).values

    # 2. MiniBatchKMeans
    mbk = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        batch_size=args.mbk_batch_size,
        n_init=args.mbk_n_init,
        max_iter=args.mbk_max_iter,
        reassignment_ratio=0.01,
    )
    y_mbk = mbk.fit_predict(X)
    row, mapping = evaluate_result(
        model_name="MiniBatchKMeans",
        y_true=y_true,
        y_pred=y_mbk,
        X_for_silhouette=X,
        extra_info={
            **common_info,
            "model_params": (
                f"batch_size={args.mbk_batch_size}, "
                f"n_init={args.mbk_n_init}, "
                f"max_iter={args.mbk_max_iter}"
            ),
            "representation": "TF-IDF",
        },
    )
    rows.append(row)
    result_df["cluster_minibatch_kmeans"] = y_mbk
    result_df["mapped_minibatch_kmeans"] = pd.Series(y_mbk).map(mapping).map(id_to_label).values

    # 3. Agglomerative Clustering
    # High-dimensional sparse TF-IDF is not ideal for Agglomerative,
    # so we first reduce dimensions by TruncatedSVD.
    svd_dim = min(args.svd_dim, X.shape[1] - 1, X.shape[0] - 1)
    svd_pipeline = make_pipeline(
        TruncatedSVD(n_components=svd_dim, random_state=args.seed),
        Normalizer(copy=False),
    )
    X_svd = svd_pipeline.fit_transform(X)

    agg = AgglomerativeClustering(
        n_clusters=args.n_clusters,
        linkage="ward",
    )
    y_agg = agg.fit_predict(X_svd)
    row, mapping = evaluate_result(
        model_name="AgglomerativeClustering",
        y_true=y_true,
        y_pred=y_agg,
        X_for_silhouette=X_svd,
        extra_info={
            **common_info,
            "model_params": f"linkage=ward, svd_dim={svd_dim}",
            "representation": "TF-IDF + TruncatedSVD + Normalizer",
        },
    )
    rows.append(row)
    result_df["cluster_agglomerative"] = y_agg
    result_df["mapped_agglomerative"] = pd.Series(y_agg).map(mapping).map(id_to_label).values

    metrics_df = pd.DataFrame(rows)
    metrics_path = output_dir / f"cluster_model_comparison_{args.tag}.csv"
    result_path = output_dir / f"cluster_result_compare_{args.tag}.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    print("===== 模型对比完成 =====")
    print(metrics_df[["model", "acc", "nmi", "ari", "silhouette_cosine"]])
    print(f"输出指标表: {metrics_path}")
    print(f"输出明细表: {result_path}")


if __name__ == "__main__":
    main()