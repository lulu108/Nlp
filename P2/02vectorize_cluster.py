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
import re
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

DEFAULT_INPUT = DATA_DIR / "news_for_cluster_440.csv"
DEFAULT_N_CLUSTERS = 4
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 50

# Stable-improvement defaults
DEFAULT_MAX_DF = 0.8
DEFAULT_MIN_DF = 4
DEFAULT_NGRAM_MIN = 1
DEFAULT_NGRAM_MAX = 1
DEFAULT_MAX_FEATURES = None
DEFAULT_SUBLINEAR_TF = True
DEFAULT_TOP_N_KEYWORDS = 15
DEFAULT_FEATURE_MODE = "word"


# Recommended presets for 440/600/800.
# 600/800 use char-level features by default, which is often more robust for Chinese short news.
PRESET_CONFIGS = {
    "440": {
        "feature_mode": "char",
        "max_df": 0.75,
        "min_df": 3,
        "ngram_min": 2,
        "ngram_max": 4,
        "max_features": 20000,
        "sublinear_tf": True,
        "n_init": 60,
        "seed": 52,
    },
    "600": {
        "feature_mode": "char",
        "max_df": 0.95,
        "min_df": 2,
        "ngram_min": 2,
        "ngram_max": 4,
        "max_features": 30000,
        "sublinear_tf": True,
        "n_init": 100,
    },
    "800": {
        "feature_mode": "char",
        "max_df": 0.95,
        "min_df": 2,
        "ngram_min": 2,
        "ngram_max": 4,
        "max_features": 40000,
        "sublinear_tf": True,
        "n_init": 120,
    },
}


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


def resolve_preset(tag: str, preset_arg: str) -> str:
    preset_arg = (preset_arg or "").strip().lower()
    tag_norm = (tag or "").strip().lower()

    if preset_arg in {"", "none", "off"}:
        return ""
    if preset_arg == "auto":
        if tag_norm in PRESET_CONFIGS:
            return tag_norm
        match = re.search(r"(440|600|800)", tag_norm)
        if match:
            return match.group(1)
        return ""
    if preset_arg in PRESET_CONFIGS:
        return preset_arg
    raise ValueError(f"Unknown preset: {preset_arg}. Available: auto/none/440/600/800")


def build_vectorizer(
    feature_mode: str,
    max_df: float,
    min_df: int,
    ngram_range: tuple[int, int],
    max_features: int | None,
    sublinear_tf: bool,
) -> TfidfVectorizer:
    vectorizer_kwargs = {
        "max_df": max_df,
        "min_df": min_df,
        "ngram_range": ngram_range,
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
        "analyzer": feature_mode,
    }
    if feature_mode == "word":
        vectorizer_kwargs["token_pattern"] = r"(?u)\b\w+\b"
    return TfidfVectorizer(**vectorizer_kwargs)


def run_one_eval(
    texts: pd.Series,
    y_true: np.ndarray,
    n_clusters: int,
    seed: int,
    n_init: int,
    feature_mode: str,
    max_df: float,
    min_df: int,
    ngram_range: tuple[int, int],
    max_features: int | None,
    sublinear_tf: bool,
    compute_silhouette: bool = True,
):
    vectorizer = build_vectorizer(
        feature_mode=feature_mode,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    y_pred = kmeans.fit_predict(X)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc, cluster_to_label_map = clustering_accuracy(y_true, y_pred)
    sil = silhouette_score(X, y_pred, metric="cosine") if compute_silhouette else 0.0

    return {
        "X": X,
        "kmeans": kmeans,
        "y_pred": y_pred,
        "vectorizer": vectorizer,
        "feature_names": vectorizer.get_feature_names_out(),
        "cluster_to_label_map": cluster_to_label_map,
        "nmi": float(nmi),
        "ari": float(ari),
        "acc": float(acc),
        "silhouette_cosine": float(sil),
        "vocab_size": int(X.shape[1]),
    }


def build_small_grid(tag_hint: str, base_seed: int) -> list[dict]:
    tag_hint = (tag_hint or "").lower()
    target = "default"
    m = re.search(r"(440|600|800)", tag_hint)
    if m:
        target = m.group(1)

    if target == "440":
        feature_modes = ["word", "char"]
        max_dfs = [0.75, 0.8]
        min_dfs = [3, 4]
        ngram_ranges = [(1, 1), (1, 2), (2, 4)]
        max_features_by_mode = {
            "word": [None],
            "char": [20000],
        }
        n_inits = [80]
        seeds = sorted(set([base_seed, 41, 52]))
    elif target in {"600", "800"}:
        feature_modes = ["word", "char"]
        max_dfs = [0.8, 0.95]
        min_dfs = [2, 4]
        ngram_ranges = [(1, 1), (2, 4)]
        max_features_by_mode = {
            "word": [None],
            "char": [30000 if target == "600" else 40000],
        }
        n_inits = [80, 100]
        seeds = [base_seed]
    else:
        feature_modes = ["word", "char"]
        max_dfs = [0.8, 0.95]
        min_dfs = [2, 4]
        ngram_ranges = [(1, 1), (2, 4)]
        max_features_by_mode = {
            "word": [None],
            "char": [30000],
        }
        n_inits = [80]
        seeds = [base_seed]

    grid = []
    for feature_mode, max_df, min_df, ngram_range, n_init, seed in product(
        feature_modes,
        max_dfs,
        min_dfs,
        ngram_ranges,
        n_inits,
        seeds,
    ):
        if feature_mode == "word" and ngram_range[0] >= 2:
            continue
        if feature_mode == "char" and ngram_range[0] == 1:
            continue
        if feature_mode == "char" and (max_df != 0.8 or min_df != 3 or ngram_range != (2, 4) or seed != base_seed):
            continue
        if feature_mode == "word" and ngram_range == (1, 2) and min_df == 4:
            continue
        for max_features in max_features_by_mode[feature_mode]:
            grid.append(
                {
                    "feature_mode": feature_mode,
                    "max_df": max_df,
                    "min_df": min_df,
                    "ngram_min": ngram_range[0],
                    "ngram_max": ngram_range[1],
                    "max_features": max_features,
                    "sublinear_tf": True,
                    "n_init": n_init,
                    "seed": seed,
                }
            )
    return grid


def parse_ngram_range(text: str) -> tuple[int, int]:
    match = re.match(r"^\((\d+)\s*,\s*(\d+)\)$", text.strip())
    if not match:
        raise ValueError(f"Invalid ngram range text: {text}")
    return int(match.group(1)), int(match.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="input CSV for clustering")
    parser.add_argument("--output-dir", default=str(DATA_DIR), help="output directory")
    parser.add_argument("--output", default="", help="override cluster result csv path")
    parser.add_argument("--tag", default="", help="output suffix tag, e.g. 440/600/800")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="random seed")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS, help="number of clusters")
    parser.add_argument("--n-init", type=int, default=DEFAULT_N_INIT, help="kmeans n_init")
    parser.add_argument(
        "--feature-mode",
        choices=["word", "char"],
        default=DEFAULT_FEATURE_MODE,
        help="tfidf feature mode: word or char",
    )
    parser.add_argument("--ngram-min", type=int, default=DEFAULT_NGRAM_MIN, help="tfidf ngram min")
    parser.add_argument("--max-df", type=float, default=DEFAULT_MAX_DF, help="tfidf max_df")
    parser.add_argument("--min-df", type=int, default=DEFAULT_MIN_DF, help="tfidf min_df")
    parser.add_argument("--ngram-max", type=int, default=DEFAULT_NGRAM_MAX, help="tfidf ngram max")
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
    parser.add_argument(
        "--preset",
        default="auto",
        help="parameter preset: auto/none/440/600/800",
    )
    parser.add_argument(
        "--grid-search",
        default="false",
        help="run small grid search and auto-apply best config (true/false)",
    )
    parser.add_argument(
        "--grid-output",
        default="",
        help="optional output csv path for grid search results",
    )
    args = parser.parse_args()

    applied_preset = resolve_preset(tag=args.tag, preset_arg=args.preset)

    feature_mode = args.feature_mode
    max_df = args.max_df
    min_df = args.min_df
    ngram_min = args.ngram_min
    ngram_max = args.ngram_max
    max_features = None if args.max_features < 0 else args.max_features
    sublinear_tf = parse_bool(args.sublinear_tf)
    n_init = args.n_init

    if applied_preset:
        cfg = PRESET_CONFIGS[applied_preset]
        feature_mode = cfg["feature_mode"]
        max_df = cfg["max_df"]
        min_df = cfg["min_df"]
        ngram_min = cfg["ngram_min"]
        ngram_max = cfg["ngram_max"]
        max_features = cfg["max_features"]
        sublinear_tf = cfg["sublinear_tf"]
        n_init = cfg["n_init"]
        if "seed" in cfg:
            args.seed = int(cfg["seed"])

    grid_search = parse_bool(args.grid_search)

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

    print("\n===== 特征参数 =====")
    print(f"preset      : {applied_preset if applied_preset else 'none'}")
    print(f"feature_mode: {feature_mode}")
    print(f"max_df      : {max_df}")
    print(f"min_df      : {min_df}")
    print(f"ngram_range : ({ngram_min}, {ngram_max})")
    print(f"max_features: {max_features}")
    print(f"sublinear_tf: {sublinear_tf}")
    print(f"n_init      : {n_init}")

    id_col = find_column(df.columns, ["id", "idx", "index"])
    label_col = find_column(df.columns, ["label", "category", "class", "tag", "type"])
    if feature_mode == "char":
        text_col = find_column(df.columns, ["text", "content", "text_cut"])
    else:
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
    y_true = df["label_id"].values

    print("\n标签映射:")
    for label, idx in label_to_id.items():
        print(f"- {label} -> {idx}")
    with outs["labelmap"].open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=2)

    if ngram_min < 1 or ngram_max < 1 or ngram_min > ngram_max:
        raise ValueError(f"Invalid ngram range: ({ngram_min}, {ngram_max})")
    ngram_range = (ngram_min, ngram_max)

    if grid_search:
        grid = build_small_grid(tag_hint=args.tag, base_seed=args.seed)
        grid_rows = []
        for idx, cfg in enumerate(grid, start=1):
            cfg_text_col = find_column(df.columns, ["text", "content", "text_cut"]) if cfg["feature_mode"] == "char" else find_column(df.columns, ["text_cut", "text", "content"])
            if cfg_text_col is None:
                continue
            eval_result = run_one_eval(
                texts=df[cfg_text_col],
                y_true=y_true,
                n_clusters=args.n_clusters,
                seed=cfg["seed"],
                n_init=cfg["n_init"],
                feature_mode=cfg["feature_mode"],
                max_df=cfg["max_df"],
                min_df=cfg["min_df"],
                ngram_range=(cfg["ngram_min"], cfg["ngram_max"]),
                max_features=cfg["max_features"],
                sublinear_tf=cfg["sublinear_tf"],
                compute_silhouette=False,
            )
            grid_rows.append(
                {
                    "rank_id": idx,
                    "feature_mode": cfg["feature_mode"],
                    "max_df": cfg["max_df"],
                    "min_df": cfg["min_df"],
                    "ngram_range": str((cfg["ngram_min"], cfg["ngram_max"])),
                    "max_features": cfg["max_features"],
                    "sublinear_tf": cfg["sublinear_tf"],
                    "n_init": cfg["n_init"],
                    "seed": cfg["seed"],
                    "vocab_size": eval_result["vocab_size"],
                    "nmi": round(eval_result["nmi"], 6),
                    "ari": round(eval_result["ari"], 6),
                    "acc": round(eval_result["acc"], 6),
                    "silhouette_cosine": round(eval_result["silhouette_cosine"], 6),
                }
            )

        if not grid_rows:
            raise RuntimeError("Grid search produced no valid candidates.")

        grid_df = pd.DataFrame(grid_rows).sort_values(
            by=["acc", "nmi", "ari", "silhouette_cosine"],
            ascending=[False, False, False, False],
        )
        grid_output = Path(args.grid_output) if args.grid_output else output_dir / f"grid_search_kmeans_{args.tag or 'run'}.csv"
        grid_df.to_csv(grid_output, index=False, encoding="utf-8-sig")

        best = grid_df.iloc[0]
        feature_mode = str(best["feature_mode"])
        max_df = float(best["max_df"])
        min_df = int(best["min_df"])
        ngram_min, ngram_max = parse_ngram_range(str(best["ngram_range"]))
        max_features = None if pd.isna(best["max_features"]) else int(best["max_features"])
        sublinear_tf = parse_bool(str(best["sublinear_tf"]))
        n_init = int(best["n_init"])
        args.seed = int(best["seed"])
        ngram_range = (ngram_min, ngram_max)
        text_col = find_column(df.columns, ["text", "content", "text_cut"]) if feature_mode == "char" else find_column(df.columns, ["text_cut", "text", "content"])
        applied_preset = f"grid_best({applied_preset or 'none'})"

        best_cfg_path = output_dir / f"grid_search_best_{args.tag or 'run'}.json"
        with best_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_mode": feature_mode,
                    "max_df": max_df,
                    "min_df": min_df,
                    "ngram_min": ngram_min,
                    "ngram_max": ngram_max,
                    "max_features": max_features,
                    "sublinear_tf": sublinear_tf,
                    "n_init": n_init,
                    "seed": args.seed,
                    "best_acc": float(best["acc"]),
                    "best_nmi": float(best["nmi"]),
                    "best_ari": float(best["ari"]),
                    "best_silhouette_cosine": float(best["silhouette_cosine"]),
                    "grid_csv": str(grid_output),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\n===== 小范围网格搜索完成 =====")
        print(f"候选数: {len(grid_df)}")
        print(f"best ACC: {best['acc']:.4f}")
        print(f"best config: {best_cfg_path}")
        print(f"grid table: {grid_output}")

    print("\n===== 最终使用参数 =====")
    print(f"preset      : {applied_preset if applied_preset else 'none'}")
    print(f"feature_mode: {feature_mode}")
    print(f"max_df      : {max_df}")
    print(f"min_df      : {min_df}")
    print(f"ngram_range : {ngram_range}")
    print(f"max_features: {max_features}")
    print(f"sublinear_tf: {sublinear_tf}")
    print(f"n_init      : {n_init}")
    print(f"seed        : {args.seed}")

    eval_result = run_one_eval(
        texts=df[text_col],
        y_true=y_true,
        n_clusters=args.n_clusters,
        seed=args.seed,
        n_init=n_init,
        feature_mode=feature_mode,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )
    X = eval_result["X"]
    kmeans = eval_result["kmeans"]
    feature_names = eval_result["feature_names"]
    cluster_to_label_map = eval_result["cluster_to_label_map"]
    nmi = eval_result["nmi"]
    ari = eval_result["ari"]
    acc = eval_result["acc"]
    sil = eval_result["silhouette_cosine"]
    df["cluster"] = eval_result["y_pred"]

    print("\n===== TF-IDF 统计 =====")
    print(f"样本数: {X.shape[0]}")
    print(f"特征维度(词表大小): {X.shape[1]}")
    print(f"稀疏矩阵形状: {X.shape}")

    y_pred = df["cluster"].values

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
        "preset": applied_preset or "none",
        "feature_mode": feature_mode,
        "max_df": max_df,
        "min_df": min_df,
        "ngram_range": str(ngram_range),
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
        "n_init": n_init,
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
