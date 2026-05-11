# -*- coding: utf-8 -*-
"""
KMeans feature-parameter grid search for P2.

Goal:
    Search TF-IDF feature parameters for original TF-IDF + KMeans.

Default input:
    P2/data/news_for_cluster_800.csv

Default outputs:
    P2/data/grid_search/kmeans_features_800/
        kmeans_feature_grid_800.csv
        kmeans_feature_grid_top_800.csv
        kmeans_feature_grid_best_800.json
        cluster_result_kmeans_feature_best_800.csv
        cluster_crosstab_kmeans_feature_best_800.csv
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


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
    """
    Example:
        "2-3,2-4,3-5" -> [(2,3), (2,4), (3,5)]
    """
    ranges: list[tuple[int, int]] = []
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
    """
    Calculate clustering ACC with Hungarian matching.
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
    X,
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
                sil = silhouette_score(X, y_pred, metric="cosine")
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


def run_single_kmeans(
    texts: pd.Series,
    y_true: np.ndarray,
    *,
    feature_mode: str,
    max_df: float,
    min_df: int,
    ngram_range: tuple[int, int],
    max_features: int | None,
    sublinear_tf: bool,
    n_clusters: int,
    seed: int,
    n_init: int,
    skip_silhouette: bool,
):
    vectorizer = build_vectorizer(
        feature_mode=feature_mode,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )

    start = time.perf_counter()
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=n_init,
    )
    y_pred = kmeans.fit_predict(X)
    elapsed = time.perf_counter() - start

    metrics = evaluate_clustering(
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        skip_silhouette=skip_silhouette,
    )

    return {
        "X": X,
        "y_pred": y_pred,
        "kmeans": kmeans,
        "vectorizer": vectorizer,
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
    parser.add_argument("--ngram-ranges", default="2-3,2-4,3-5")
    parser.add_argument("--min-dfs", default="1,2,3")
    parser.add_argument("--max-dfs", default="0.85,0.90,0.95,1.0")
    parser.add_argument("--max-features-list", default="30000,40000,60000,80000")
    parser.add_argument("--seeds", default="21,42,52,66,100")
    parser.add_argument("--n-init", type=int, default=120)
    parser.add_argument("--sublinear-tf", default="true")
    parser.add_argument("--skip-silhouette", default="true")
    parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_root)
    output_dir = Path(args.output_dir) if args.output_dir else output_root / f"kmeans_features_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ngram_ranges = parse_ngram_ranges(args.ngram_ranges)
    min_dfs = parse_int_list(args.min_dfs)
    max_dfs = parse_float_list(args.max_dfs)
    max_features_list = parse_int_list(args.max_features_list)
    seeds = parse_int_list(args.seeds)
    sublinear_tf = parse_bool(args.sublinear_tf)
    skip_silhouette = parse_bool(args.skip_silhouette)

    print("===== KMeans feature grid search =====", flush=True)
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

    total = (
        len(ngram_ranges)
        * len(min_dfs)
        * len(max_dfs)
        * len(max_features_list)
        * len(seeds)
    )

    rows = []
    best_cache = None
    best_key = None

    for idx, (ngram_range, min_df, max_df, max_features, seed) in enumerate(
        product(ngram_ranges, min_dfs, max_dfs, max_features_list, seeds),
        start=1,
    ):
        print(
            f"===== [{idx}/{total}] "
            f"ngram={ngram_range}, min_df={min_df}, max_df={max_df}, "
            f"max_features={max_features}, seed={seed} =====",
            flush=True,
        )

        try:
            result = run_single_kmeans(
                texts=df[text_col],
                y_true=y_true,
                feature_mode=args.feature_mode,
                max_df=max_df,
                min_df=min_df,
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=sublinear_tf,
                n_clusters=args.n_clusters,
                seed=seed,
                n_init=args.n_init,
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
                "seed": seed,
                "n_init": args.n_init,
                "n_samples": len(df),
                "n_clusters": args.n_clusters,
                "vocab_size": int(result["X"].shape[1]),
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
                    "seed": seed,
                    "n_init": args.n_init,
                    "error": str(exc),
                }
            )

    grid_df = pd.DataFrame(rows).sort_values(
        by=["acc", "nmi", "ari", "silhouette_cosine"],
        ascending=[False, False, False, False],
        na_position="last",
    )

    grid_path = output_dir / f"kmeans_feature_grid_{args.tag}.csv"
    top_path = output_dir / f"kmeans_feature_grid_top_{args.tag}.csv"
    best_json_path = output_dir / f"kmeans_feature_grid_best_{args.tag}.json"
    result_path = output_dir / f"cluster_result_kmeans_feature_best_{args.tag}.csv"
    crosstab_path = output_dir / f"cluster_crosstab_kmeans_feature_best_{args.tag}.csv"

    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    grid_df.head(args.top_k).to_csv(top_path, index=False, encoding="utf-8-sig")

    if best_cache is None:
        raise RuntimeError("No successful KMeans grid result.")

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

    print("\n===== KMeans feature grid search done =====", flush=True)
    print("Best result:", flush=True)
    print(pd.DataFrame([best_row])[["acc", "nmi", "ari", "silhouette_cosine", "ngram_range", "min_df", "max_df", "max_features", "seed"]], flush=True)
    print(f"grid csv      : {grid_path}", flush=True)
    print(f"top csv       : {top_path}", flush=True)
    print(f"best json     : {best_json_path}", flush=True)
    print(f"best result   : {result_path}", flush=True)
    print(f"best crosstab : {crosstab_path}", flush=True)


if __name__ == "__main__":
    main()