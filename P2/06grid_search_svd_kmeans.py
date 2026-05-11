# -*- coding: utf-8 -*-
"""
SVD + KMeans grid search for P2.

Goal:
    Test TF-IDF -> TruncatedSVD -> Normalizer -> KMeans.

Default input:
    P2/data/news_for_cluster_800.csv

Default outputs:
    P2/data/grid_search/svd_kmeans_800/
        svd_kmeans_grid_800.csv
        svd_kmeans_grid_top_800.csv
        svd_kmeans_grid_best_800.json
        cluster_result_svd_kmeans_best_800.csv
        cluster_crosstab_svd_kmeans_best_800.csv
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "news_for_cluster_800.csv"
DEFAULT_OUTPUT_ROOT = DATA_DIR / "grid_search"


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_ngram_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    for seg in str(text).split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "-" not in seg:
            raise ValueError(f"Invalid ngram range: {seg}. Expected like 2-4.")
        left, right = seg.split("-", 1)
        a, b = int(left.strip()), int(right.strip())
        if a <= 0 or b <= 0 or a > b:
            raise ValueError(f"Invalid ngram range: {seg}")
        ranges.append((a, b))
    if not ranges:
        raise ValueError("No valid ngram ranges parsed.")
    return ranges


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
        cost_matrix[pred_to_idx[yp], true_to_idx[yt]] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    matched = cost_matrix[row_ind, col_ind].sum()
    acc = matched / len(y_true)

    idx_to_true = {i: label for label, i in true_to_idx.items()}
    idx_to_pred = {i: label for label, i in pred_to_idx.items()}

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r in idx_to_pred and c in idx_to_true:
            mapping[int(idx_to_pred[r])] = int(idx_to_true[c])

    return float(acc), mapping


def build_vectorizer(
    feature_mode: str,
    max_df: float,
    min_df: int,
    ngram_range: tuple[int, int],
    max_features: int | None,
    sublinear_tf: bool,
):
    kwargs = {
        "analyzer": feature_mode,
        "max_df": max_df,
        "min_df": min_df,
        "ngram_range": ngram_range,
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
    }

    if feature_mode == "word":
        kwargs["token_pattern"] = r"(?u)\b\w+\b"

    return TfidfVectorizer(**kwargs)


def evaluate_clustering(
    X_for_silhouette,
    y_true,
    y_pred,
    skip_silhouette: bool,
):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc, mapping = clustering_accuracy(y_true, y_pred)

    if skip_silhouette:
        sil = np.nan
    else:
        unique_pred = np.unique(y_pred)
        if 1 < len(unique_pred) < len(y_pred):
            try:
                sil = silhouette_score(X_for_silhouette, y_pred, metric="cosine")
            except Exception as exc:
                print(f"[WARN] silhouette_score failed: {exc}", flush=True)
                sil = np.nan
        else:
            sil = np.nan

    return {
        "acc": float(acc),
        "nmi": float(nmi),
        "ari": float(ari),
        "silhouette_cosine": float(sil) if not np.isnan(sil) else np.nan,
        "cluster_to_label_mapping": mapping,
    }


def run_single_svd_kmeans(
    X_tfidf,
    y_true: np.ndarray,
    *,
    svd_dim: int,
    n_clusters: int,
    seed: int,
    n_init: int,
    skip_silhouette: bool,
):
    safe_dim = min(svd_dim, X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1)
    if safe_dim < 2:
        raise ValueError(f"Invalid svd_dim after clipping: {safe_dim}")

    start = time.perf_counter()

    reducer = make_pipeline(
        TruncatedSVD(n_components=safe_dim, random_state=seed),
        Normalizer(copy=False),
    )
    X_svd = reducer.fit_transform(X_tfidf)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=n_init,
    )
    y_pred = kmeans.fit_predict(X_svd)

    elapsed = time.perf_counter() - start

    metrics = evaluate_clustering(
        X_for_silhouette=X_svd,
        y_true=y_true,
        y_pred=y_pred,
        skip_silhouette=skip_silhouette,
    )

    return {
        "X_svd": X_svd,
        "y_pred": y_pred,
        "kmeans": kmeans,
        "svd_dim_used": safe_dim,
        "elapsed_seconds": elapsed,
        **metrics,
    }


def make_result_and_crosstab(
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mapping: dict[int, int],
    id_to_label: dict[int, str],
    result_path: Path,
    crosstab_path: Path,
):
    result_df = df[[label_col, text_col]].copy()
    result_df["label_id"] = y_true
    result_df["cluster"] = y_pred
    result_df["mapped_label_id"] = [mapping.get(int(c), -1) for c in y_pred]
    result_df["mapped_label_name"] = [
        id_to_label.get(mapping.get(int(c), -1), "unknown") for c in y_pred
    ]

    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    ctab = pd.crosstab(result_df["cluster"], result_df[label_col])
    ctab.to_csv(crosstab_path, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--tag", default="800")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="", help="Optional exact output directory.")

    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--feature-mode", choices=["word", "char"], default="char")
    parser.add_argument("--ngram-ranges", default="2-4")
    parser.add_argument("--min-dfs", default="2")
    parser.add_argument("--max-dfs", default="0.95")
    parser.add_argument("--max-features-list", default="40000")
    parser.add_argument("--sublinear-tf", default="true")

    parser.add_argument("--svd-dims", default="50,100,150,200,300")
    parser.add_argument("--seeds", default="42,52,66")
    parser.add_argument("--n-inits", default="80,120")
    parser.add_argument("--skip-silhouette", default="false")
    parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_root)
    output_dir = Path(args.output_dir) if args.output_dir else output_root / f"svd_kmeans_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ngram_ranges = parse_ngram_ranges(args.ngram_ranges)
    min_dfs = parse_int_list(args.min_dfs)
    max_dfs = parse_float_list(args.max_dfs)
    max_features_list = parse_int_list(args.max_features_list)
    svd_dims = parse_int_list(args.svd_dims)
    seeds = parse_int_list(args.seeds)
    n_inits = parse_int_list(args.n_inits)

    sublinear_tf = parse_bool(args.sublinear_tf)
    skip_silhouette = parse_bool(args.skip_silhouette)

    print("===== SVD + KMeans grid search =====", flush=True)
    print(f"input: {input_path}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)

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

    rows = []
    best_cache = None
    best_key = None

    tfidf_configs = list(product(ngram_ranges, min_dfs, max_dfs, max_features_list))
    total = len(tfidf_configs) * len(svd_dims) * len(seeds) * len(n_inits)
    idx = 0

    for ngram_range, min_df, max_df, max_features in tfidf_configs:
        print(
            f"===== TF-IDF build: ngram={ngram_range}, min_df={min_df}, "
            f"max_df={max_df}, max_features={max_features} =====",
            flush=True,
        )

        try:
            vectorizer = build_vectorizer(
                feature_mode=args.feature_mode,
                max_df=max_df,
                min_df=min_df,
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=sublinear_tf,
            )
            X_tfidf = vectorizer.fit_transform(df[text_col])
        except Exception as exc:
            print(f"[WARN] TF-IDF failed: {exc}", flush=True)
            continue

        for svd_dim, seed, n_init in product(svd_dims, seeds, n_inits):
            idx += 1
            print(
                f"===== [{idx}/{total}] svd_dim={svd_dim}, seed={seed}, n_init={n_init} =====",
                flush=True,
            )

            try:
                result = run_single_svd_kmeans(
                    X_tfidf=X_tfidf,
                    y_true=y_true,
                    svd_dim=svd_dim,
                    n_clusters=args.n_clusters,
                    seed=seed,
                    n_init=n_init,
                    skip_silhouette=skip_silhouette,
                )

                row = {
                    "rank_id": idx,
                    "tag": args.tag,
                    "feature_mode": args.feature_mode,
                    "ngram_range": str(ngram_range),
                    "min_df": min_df,
                    "max_df": max_df,
                    "max_features": max_features,
                    "sublinear_tf": sublinear_tf,
                    "svd_dim": svd_dim,
                    "svd_dim_used": result["svd_dim_used"],
                    "seed": seed,
                    "n_init": n_init,
                    "n_samples": len(df),
                    "n_clusters": args.n_clusters,
                    "vocab_size": int(X_tfidf.shape[1]),
                    "acc": round(result["acc"], 6),
                    "nmi": round(result["nmi"], 6),
                    "ari": round(result["ari"], 6),
                    "silhouette_cosine": (
                        round(result["silhouette_cosine"], 6)
                        if not np.isnan(result["silhouette_cosine"])
                        else np.nan
                    ),
                    "elapsed_seconds": round(result["elapsed_seconds"], 4),
                    "cluster_to_label_mapping": str(result["cluster_to_label_mapping"]),
                    "representation": "TF-IDF + TruncatedSVD + Normalizer",
                }
                rows.append(row)

                key = (
                    row["acc"],
                    row["nmi"],
                    row["ari"],
                    -999 if pd.isna(row["silhouette_cosine"]) else row["silhouette_cosine"],
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best_cache = {
                        "row": row,
                        "y_pred": result["y_pred"],
                        "mapping": result["cluster_to_label_mapping"],
                    }

            except Exception as exc:
                print(f"[WARN] failed: {exc}", flush=True)
                rows.append(
                    {
                        "rank_id": idx,
                        "tag": args.tag,
                        "feature_mode": args.feature_mode,
                        "ngram_range": str(ngram_range),
                        "min_df": min_df,
                        "max_df": max_df,
                        "max_features": max_features,
                        "sublinear_tf": sublinear_tf,
                        "svd_dim": svd_dim,
                        "seed": seed,
                        "n_init": n_init,
                        "error": str(exc),
                    }
                )

    grid_df = pd.DataFrame(rows).sort_values(
        by=["acc", "nmi", "ari", "silhouette_cosine"],
        ascending=[False, False, False, False],
        na_position="last",
    )

    grid_path = output_dir / f"svd_kmeans_grid_{args.tag}.csv"
    top_path = output_dir / f"svd_kmeans_grid_top_{args.tag}.csv"
    best_json_path = output_dir / f"svd_kmeans_grid_best_{args.tag}.json"
    result_path = output_dir / f"cluster_result_svd_kmeans_best_{args.tag}.csv"
    crosstab_path = output_dir / f"cluster_crosstab_svd_kmeans_best_{args.tag}.csv"

    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    grid_df.head(args.top_k).to_csv(top_path, index=False, encoding="utf-8-sig")

    if best_cache is None:
        raise RuntimeError("No successful SVD + KMeans grid result.")

    best_row = best_cache["row"]
    with best_json_path.open("w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)

    make_result_and_crosstab(
        df=df,
        label_col=label_col,
        text_col=text_col,
        y_true=y_true,
        y_pred=best_cache["y_pred"],
        mapping=best_cache["mapping"],
        id_to_label=id_to_label,
        result_path=result_path,
        crosstab_path=crosstab_path,
    )

    print("\n===== SVD + KMeans grid search done =====", flush=True)
    print("Best result:", flush=True)
    print(pd.DataFrame([best_row])[["acc", "nmi", "ari", "silhouette_cosine", "svd_dim", "ngram_range", "min_df", "max_df", "max_features", "seed", "n_init"]], flush=True)
    print(f"grid csv      : {grid_path}", flush=True)
    print(f"top csv       : {top_path}", flush=True)
    print(f"best json     : {best_json_path}", flush=True)
    print(f"best result   : {result_path}", flush=True)
    print(f"best crosstab : {crosstab_path}", flush=True)


if __name__ == "__main__":
    main()