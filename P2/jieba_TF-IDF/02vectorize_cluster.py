# -*- coding: utf-8 -*-
"""
Stage-C clustering: TF-IDF + KMeans + evaluation.

Main outputs (with optional tag suffix):
    cluster_result_kmeans_{tag}.csv
    cluster_metrics_kmeans_{tag}.csv
    cluster_top_keywords_kmeans_{tag}.csv
    cluster_crosstab_kmeans_{tag}.csv
    label_mapping_kmeans_{tag}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_INPUT = DATA_DIR / "news_for_cluster_440.csv"
DEFAULT_N_CLUSTERS = 4
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 50

# Stable-improvement defaults
DEFAULT_MAX_DF = 0.8
DEFAULT_MIN_DF = 4
DEFAULT_NGRAM_MAX = 1
DEFAULT_MAX_FEATURES = None
DEFAULT_SUBLINEAR_TF = True
DEFAULT_TOP_N_KEYWORDS = 15


def output_paths(output_dir: Path, tag: str, result_override: str | None) -> dict[str, Path]:
    suffix = f"_{tag}" if tag else ""
    result_path = Path(result_override) if result_override else output_dir / f"cluster_result_kmeans{suffix}.csv"
    return {
        "result": result_path,
        "metrics": output_dir / f"cluster_metrics_kmeans{suffix}.csv",
        "keywords": output_dir / f"cluster_top_keywords_kmeans{suffix}.csv",
        "crosstab": output_dir / f"cluster_crosstab_kmeans{suffix}.csv",
        "labelmap": output_dir / f"label_mapping_kmeans{suffix}.json",
    }


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
    return acc, mapping


def get_top_keywords_per_cluster(kmeans_model, feature_names, top_n=10):
    centers = kmeans_model.cluster_centers_
    cluster_keywords = {}
    for cluster_id in range(centers.shape[0]):
        center = centers[cluster_id]
        top_idx = np.argsort(center)[::-1][:top_n]
        cluster_keywords[cluster_id] = [feature_names[i] for i in top_idx]
    return cluster_keywords


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="input CSV for clustering")
    parser.add_argument("--output-dir", default=str(DATA_DIR), help="output directory")
    parser.add_argument("--output", default="", help="override cluster result csv path")
    parser.add_argument("--tag", default="", help="output suffix tag, e.g. 440/600/800")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="random seed")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS, help="number of clusters")
    parser.add_argument("--n-init", type=int, default=DEFAULT_N_INIT, help="kmeans n_init")
    parser.add_argument("--max-df", type=float, default=DEFAULT_MAX_DF, help="tfidf max_df")
    parser.add_argument("--min-df", type=int, default=DEFAULT_MIN_DF, help="tfidf min_df")
    parser.add_argument("--ngram-max", type=int, default=DEFAULT_NGRAM_MAX, help="tfidf ngram max (1 or 2)")
    parser.add_argument("--max-features", type=int, default=-1, help="tfidf max_features, -1 means None")
    parser.add_argument(
        "--sublinear-tf",
        default=str(DEFAULT_SUBLINEAR_TF),
        help="tfidf sublinear_tf (true/false)",
    )
    parser.add_argument("--top-n-keywords", type=int, default=DEFAULT_TOP_N_KEYWORDS, help="top keywords per cluster")
    parser.add_argument(
        "--summary-csv",
        default="",
        help="optional summary CSV path to append current metrics row",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outs = output_paths(output_dir=output_dir, tag=args.tag, result_override=args.output or None)

    print("===== 阶段C：TF-IDF + KMeans 聚类开始 =====")
    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_file}")

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    print(f"读取文件: {input_file}")
    print(f"样本数: {len(df)}")
    print(f"列名: {list(df.columns)}")

    id_col = find_column(df.columns, ["id", "idx", "index"])
    label_col = find_column(df.columns, ["label", "category", "class", "tag", "type"])
    text_col = find_column(df.columns, ["text_cut", "text", "content"])

    if id_col is None:
        id_col = "id"
        df[id_col] = range(1, len(df) + 1)
        print("未发现 id 列，已自动创建。")
    if label_col is None:
        raise ValueError("未识别到 label 列，当前脚本需要真实标签做评估。")
    if text_col is None:
        raise ValueError("未识别到文本列，请确保有 text_cut 或 text 列。")

    before_drop = len(df)
    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.strip() != ""].copy()
    print(f"空文本去除: {before_drop - len(df)}")
    print(f"有效样本数: {len(df)}")

    label_names = sorted(df[label_col].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(label_names)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    df["label_id"] = df[label_col].map(label_to_id)

    print("\n标签映射:")
    for label, idx in label_to_id.items():
        print(f"- {label} -> {idx}")
    with outs["labelmap"].open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=2)

    max_features = None if args.max_features < 0 else args.max_features
    ngram_range = (1, args.ngram_max)
    sublinear_tf = parse_bool(args.sublinear_tf)

    vectorizer = TfidfVectorizer(
        max_df=args.max_df,
        min_df=args.min_df,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        token_pattern=r"(?u)\b\w+\b",
    )

    X = vectorizer.fit_transform(df[text_col])
    feature_names = vectorizer.get_feature_names_out()

    print("\n===== TF-IDF 统计 =====")
    print(f"样本数: {X.shape[0]}")
    print(f"特征维度(词表大小): {X.shape[1]}")
    print(f"稀疏矩阵形状: {X.shape}")

    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        n_init=args.n_init,
    )
    df["cluster"] = kmeans.fit_predict(X)

    y_true = df["label_id"].values
    y_pred = df["cluster"].values
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc, cluster_to_label_map = clustering_accuracy(y_true, y_pred)
    sil = silhouette_score(X, y_pred, metric="cosine")

    print("\n===== 聚类评估指标 =====")
    print(f"NMI         : {nmi:.4f}")
    print(f"ARI         : {ari:.4f}")
    print(f"ACC         : {acc:.4f}")
    print(f"Silhouette  : {sil:.4f}")

    print("\n簇 -> 真实标签 映射（ACC最优匹配）:")
    for cluster_id in sorted(cluster_to_label_map.keys()):
        mapped_label_id = cluster_to_label_map[cluster_id]
        mapped_label_name = id_to_label[mapped_label_id]
        print(f"- cluster {cluster_id} -> {mapped_label_id} ({mapped_label_name})")

    df["mapped_label_id"] = df["cluster"].map(cluster_to_label_map)
    df["mapped_label_name"] = df["mapped_label_id"].map(id_to_label)

    ctab = pd.crosstab(df["cluster"], df[label_col])
    ctab.to_csv(outs["crosstab"], encoding="utf-8-sig")
    print("\n===== cluster × 真实类别 分布表 =====")
    print(ctab)

    cluster_keywords = get_top_keywords_per_cluster(
        kmeans_model=kmeans,
        feature_names=feature_names,
        top_n=args.top_n_keywords,
    )

    keyword_rows = []
    print("\n===== 每个簇的Top关键词 =====")
    for cluster_id in range(args.n_clusters):
        keywords = cluster_keywords.get(cluster_id, [])
        mapped_label_id = cluster_to_label_map.get(cluster_id, None)
        mapped_label_name = id_to_label.get(mapped_label_id, "unknown") if mapped_label_id is not None else "unknown"
        print(f"cluster {cluster_id} -> {mapped_label_name}: {' / '.join(keywords)}")
        keyword_rows.append(
            {
                "cluster": cluster_id,
                "mapped_label_id": mapped_label_id,
                "mapped_label_name": mapped_label_name,
                "top_keywords": " ".join(keywords),
            }
        )

    result_cols = [id_col, label_col, "label_id", text_col, "cluster", "mapped_label_id", "mapped_label_name"]
    result_cols = [c for c in result_cols if c in df.columns]
    df[result_cols].to_csv(outs["result"], index=False, encoding="utf-8-sig")

    metrics_row = {
        "tag": args.tag,
        "seed": args.seed,
        "n_samples": X.shape[0],
        "n_clusters": args.n_clusters,
        "vocab_size": X.shape[1],
        "max_df": args.max_df,
        "min_df": args.min_df,
        "ngram_range": str(ngram_range),
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
        "n_init": args.n_init,
        "nmi": round(float(nmi), 6),
        "ari": round(float(ari), 6),
        "acc": round(float(acc), 6),
        "silhouette_cosine": round(float(sil), 6),
    }
    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(outs["metrics"], index=False, encoding="utf-8-sig")
    pd.DataFrame(keyword_rows).to_csv(outs["keywords"], index=False, encoding="utf-8-sig")

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if summary_path.exists():
            old = pd.read_csv(summary_path, encoding="utf-8-sig")
            merged = pd.concat([old, pd.DataFrame([metrics_row])], ignore_index=True)
        else:
            merged = pd.DataFrame([metrics_row])
        merged.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n输出文件:")
    for k in ("result", "metrics", "keywords", "crosstab", "labelmap"):
        print(f"- {outs[k]}")
    if args.summary_csv:
        print(f"- {Path(args.summary_csv)}")
    print("\n===== 阶段C完成 =====")


if __name__ == "__main__":
    main()
