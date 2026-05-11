# -*- coding: utf-8 -*-
"""
Compare clustering models for P2:
KMeans vs MiniBatchKMeans vs Agglomerative Clustering.

Outputs:
    P2/data/cluster_model_comparison_{tag}.csv
    P2/data/cluster_result_compare_{tag}.csv
    P2/data/minibatch_kmeans_grid_{tag}.csv
    P2/data/cluster_crosstab_compare_{tag}_KMeans.csv
    P2/data/cluster_crosstab_compare_{tag}_MiniBatchKMeans.csv
    P2/data/cluster_crosstab_compare_{tag}_AgglomerativeClustering.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "news_for_cluster_800.csv"


def parse_bool(value: str) -> bool:
    value = str(value).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


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


def safe_silhouette(X, y_pred, skip_silhouette: bool, metric: str = "cosine"):
    if skip_silhouette:
        return np.nan

    unique_pred = np.unique(y_pred)
    if not (1 < len(unique_pred) < len(y_pred)):
        return np.nan

    try:
        return float(silhouette_score(X, y_pred, metric=metric))
    except Exception as exc:
        print(f"[WARN] silhouette_score 计算失败: {exc}", flush=True)
        return np.nan


def evaluate_result(
    model_name: str,
    y_true,
    y_pred,
    X_for_silhouette,
    skip_silhouette: bool,
    silhouette_metric: str = "cosine",
    extra_info: dict | None = None,
):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc, mapping = clustering_accuracy(y_true, y_pred)
    sil = safe_silhouette(
        X_for_silhouette,
        y_pred,
        skip_silhouette=skip_silhouette,
        metric=silhouette_metric,
    )

    row = {
        "model": model_name,
        "n_clusters_pred": len(np.unique(y_pred)),
        "acc": round(float(acc), 6),
        "nmi": round(float(nmi), 6),
        "ari": round(float(ari), 6),
        "silhouette_cosine": round(float(sil), 6) if not np.isnan(sil) else np.nan,
        "cluster_to_label_mapping": str(mapping),
    }

    if extra_info:
        row.update(extra_info)

    return row, mapping


def make_mapped_label(y_pred, mapping, id_to_label):
    mapped = []
    for c in y_pred:
        label_id = mapping.get(c, None)
        mapped.append(id_to_label.get(label_id, "unknown"))
    return mapped


def save_crosstab(result_df, label_col, cluster_col, output_path):
    ctab = pd.crosstab(result_df[cluster_col], result_df[label_col])
    ctab.to_csv(output_path, encoding="utf-8-sig")
    return ctab


def run_minibatch_grid(
    X,
    y_true,
    n_clusters: int,
    seed: int,
    skip_silhouette: bool,
    batch_sizes: list[int],
    n_inits: list[int],
    max_iters: list[int],
):
    rows = []
    best = None

    total = len(batch_sizes) * len(n_inits) * len(max_iters)
    idx = 0

    for batch_size in batch_sizes:
        for n_init in n_inits:
            for max_iter in max_iters:
                idx += 1
                print(
                    f"===== MiniBatchKMeans grid {idx}/{total}: "
                    f"batch_size={batch_size}, n_init={n_init}, max_iter={max_iter} =====",
                    flush=True,
                )

                start = time.perf_counter()
                model = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=seed,
                    batch_size=batch_size,
                    n_init=n_init,
                    max_iter=max_iter,
                    reassignment_ratio=0.0,
                    init="k-means++",
                    compute_labels=True,
                )
                y_pred = model.fit_predict(X)
                elapsed = time.perf_counter() - start

                row, mapping = evaluate_result(
                    model_name="MiniBatchKMeans",
                    y_true=y_true,
                    y_pred=y_pred,
                    X_for_silhouette=X,
                    skip_silhouette=skip_silhouette,
                    silhouette_metric="cosine",
                    extra_info={
                        "batch_size": batch_size,
                        "n_init": n_init,
                        "max_iter": max_iter,
                        "reassignment_ratio": 0.0,
                        "elapsed_seconds": round(elapsed, 4),
                    },
                )

                row["_y_pred"] = y_pred
                row["_mapping"] = mapping
                rows.append(row)

                if best is None:
                    best = row
                else:
                    # Main selection by ACC, then NMI, then ARI.
                    key_now = (row["acc"], row["nmi"], row["ari"])
                    key_best = (best["acc"], best["nmi"], best["ari"])
                    if key_now > key_best:
                        best = row

    grid_df = pd.DataFrame(
        [
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in rows
        ]
    ).sort_values(
        by=["acc", "nmi", "ari", "silhouette_cosine"],
        ascending=[False, False, False, False],
    )

    return best, grid_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--tag", default="800")
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=4)

    parser.add_argument("--feature-mode", choices=["word", "char"], default="char")
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--sublinear-tf", default="true")

    parser.add_argument("--kmeans-n-init", type=int, default=120)

    # MiniBatchKMeans grid options.
    parser.add_argument("--mbk-batch-sizes", default="64,128,256,512,800")
    parser.add_argument("--mbk-n-inits", default="20,50,100")
    parser.add_argument("--mbk-max-iters", default="300,500,1000")

    parser.add_argument("--svd-dim", type=int, default=100)
    parser.add_argument("--skip-silhouette", default="false")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sublinear_tf = parse_bool(args.sublinear_tf)
    skip_silhouette = parse_bool(args.skip_silhouette)
    max_features = None if args.max_features <= 0 else args.max_features

    mbk_batch_sizes = [int(x.strip()) for x in args.mbk_batch_sizes.split(",") if x.strip()]
    mbk_n_inits = [int(x.strip()) for x in args.mbk_n_inits.split(",") if x.strip()]
    mbk_max_iters = [int(x.strip()) for x in args.mbk_max_iters.split(",") if x.strip()]

    print("===== 读取数据开始 =====", flush=True)
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"读取文件: {input_path}", flush=True)
    print(f"原始样本数: {len(df)}", flush=True)
    print(f"列名: {list(df.columns)}", flush=True)

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
    print(f"有效样本数: {len(df)}", flush=True)

    label_names = sorted(df[label_col].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(label_names)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    y_true = df[label_col].map(label_to_id).values

    print("标签映射:", label_to_id, flush=True)

    print("===== TF-IDF 向量化开始 =====", flush=True)
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
    print(f"TF-IDF shape: {X.shape}", flush=True)
    print("===== TF-IDF 向量化完成 =====", flush=True)

    common_info = {
        "tag": args.tag,
        "n_samples": X.shape[0],
        "n_clusters": args.n_clusters,
        "feature_mode": args.feature_mode,
        "vocab_size": X.shape[1],
        "max_df": args.max_df,
        "min_df": args.min_df,
        "ngram_range": f"({args.ngram_min}, {args.ngram_max})",
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
    }

    rows = []
    result_df = df[[label_col, text_col]].copy()

    # 1. KMeans
    print("===== KMeans 开始 =====", flush=True)
    start = time.perf_counter()
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        n_init=args.kmeans_n_init,
    )
    y_kmeans = kmeans.fit_predict(X)
    elapsed = time.perf_counter() - start
    print(f"===== KMeans 完成，用时 {elapsed:.2f}s =====", flush=True)

    row, mapping = evaluate_result(
        model_name="KMeans",
        y_true=y_true,
        y_pred=y_kmeans,
        X_for_silhouette=X,
        skip_silhouette=skip_silhouette,
        silhouette_metric="cosine",
        extra_info={
            **common_info,
            "representation": "TF-IDF",
            "model_params": f"n_init={args.kmeans_n_init}",
            "elapsed_seconds": round(elapsed, 4),
        },
    )
    rows.append(row)
    result_df["cluster_kmeans"] = y_kmeans
    result_df["mapped_kmeans"] = make_mapped_label(y_kmeans, mapping, id_to_label)

    # 2. MiniBatchKMeans grid search
    print("===== MiniBatchKMeans 参数搜索开始 =====", flush=True)
    best_mbk, mbk_grid_df = run_minibatch_grid(
        X=X,
        y_true=y_true,
        n_clusters=args.n_clusters,
        seed=args.seed,
        skip_silhouette=skip_silhouette,
        batch_sizes=mbk_batch_sizes,
        n_inits=mbk_n_inits,
        max_iters=mbk_max_iters,
    )

    y_mbk = best_mbk["_y_pred"]
    mapping = best_mbk["_mapping"]

    best_mbk_row = {
        k: v for k, v in best_mbk.items()
        if not k.startswith("_")
    }
    best_mbk_row.update(
        {
            **common_info,
            "representation": "TF-IDF",
            "model_params": (
                f"batch_size={best_mbk['batch_size']}, "
                f"n_init={best_mbk['n_init']}, "
                f"max_iter={best_mbk['max_iter']}, "
                f"reassignment_ratio=0.0"
            ),
        }
    )
    rows.append(best_mbk_row)

    print(
        "===== MiniBatchKMeans 最优结果: "
        f"ACC={best_mbk['acc']}, NMI={best_mbk['nmi']}, ARI={best_mbk['ari']} =====",
        flush=True,
    )

    result_df["cluster_minibatch_kmeans"] = y_mbk
    result_df["mapped_minibatch_kmeans"] = make_mapped_label(y_mbk, mapping, id_to_label)

    # 3. Agglomerative Clustering
    print("===== SVD 降维开始 =====", flush=True)
    svd_dim = min(args.svd_dim, X.shape[1] - 1, X.shape[0] - 1)
    svd_pipeline = make_pipeline(
        TruncatedSVD(n_components=svd_dim, random_state=args.seed),
        Normalizer(copy=False),
    )
    X_svd = svd_pipeline.fit_transform(X)
    print(f"SVD shape: {X_svd.shape}", flush=True)
    print("===== SVD 降维完成 =====", flush=True)

    print("===== AgglomerativeClustering 开始 =====", flush=True)
    start = time.perf_counter()
    agg = AgglomerativeClustering(
        n_clusters=args.n_clusters,
        linkage="ward",
    )
    y_agg = agg.fit_predict(X_svd)
    elapsed = time.perf_counter() - start
    print(f"===== AgglomerativeClustering 完成，用时 {elapsed:.2f}s =====", flush=True)

    row, mapping = evaluate_result(
        model_name="AgglomerativeClustering",
        y_true=y_true,
        y_pred=y_agg,
        X_for_silhouette=X_svd,
        skip_silhouette=skip_silhouette,
        silhouette_metric="cosine",
        extra_info={
            **common_info,
            "representation": "TF-IDF + TruncatedSVD + Normalizer",
            "model_params": f"linkage=ward, svd_dim={svd_dim}",
            "elapsed_seconds": round(elapsed, 4),
        },
    )
    rows.append(row)
    result_df["cluster_agglomerative"] = y_agg
    result_df["mapped_agglomerative"] = make_mapped_label(y_agg, mapping, id_to_label)

    # Save outputs
    metrics_df = pd.DataFrame(rows)
    metrics_path = output_dir / f"cluster_model_comparison_{args.tag}.csv"
    result_path = output_dir / f"cluster_result_compare_{args.tag}.csv"
    mbk_grid_path = output_dir / f"minibatch_kmeans_grid_{args.tag}.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    mbk_grid_df.to_csv(mbk_grid_path, index=False, encoding="utf-8-sig")

    crosstab_paths = {
        "KMeans": output_dir / f"cluster_crosstab_compare_{args.tag}_KMeans.csv",
        "MiniBatchKMeans": output_dir / f"cluster_crosstab_compare_{args.tag}_MiniBatchKMeans.csv",
        "AgglomerativeClustering": output_dir / f"cluster_crosstab_compare_{args.tag}_AgglomerativeClustering.csv",
    }

    save_crosstab(result_df, label_col, "cluster_kmeans", crosstab_paths["KMeans"])
    save_crosstab(result_df, label_col, "cluster_minibatch_kmeans", crosstab_paths["MiniBatchKMeans"])
    save_crosstab(result_df, label_col, "cluster_agglomerative", crosstab_paths["AgglomerativeClustering"])

    print("\n===== 模型对比完成 =====", flush=True)
    print(metrics_df[["model", "acc", "nmi", "ari", "silhouette_cosine", "elapsed_seconds"]], flush=True)
    print(f"输出指标表: {metrics_path}", flush=True)
    print(f"输出明细表: {result_path}", flush=True)
    print(f"输出 MiniBatchKMeans 搜索表: {mbk_grid_path}", flush=True)
    for name, path in crosstab_paths.items():
        print(f"输出交叉表 {name}: {path}", flush=True)


if __name__ == "__main__":
    main()