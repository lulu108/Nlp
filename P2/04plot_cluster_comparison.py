# -*- coding: utf-8 -*-
"""
Plot clustering comparison figures for P2.

Inputs:
    P2/data/news_for_cluster_800.csv
    P2/data/cluster_model_comparison_800.csv
    P2/data/cluster_result_compare_800.csv
    P2/data/minibatch_kmeans_grid_800.csv

Outputs:
    P2/figures/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"


def setup_chinese_font():
    """
    Try common Chinese fonts. If unavailable, matplotlib will fall back automatically.
    """
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"保存图像: {path}")


def plot_metrics_grouped_bar(metrics_df: pd.DataFrame, output_path: Path):
    metric_cols = ["acc", "nmi", "ari", "silhouette_cosine"]
    show_cols = [c for c in metric_cols if c in metrics_df.columns]

    models = metrics_df["model"].astype(str).tolist()
    x = np.arange(len(models))
    width = 0.18

    plt.figure(figsize=(10, 5.5))

    for i, metric in enumerate(show_cols):
        values = metrics_df[metric].astype(float).fillna(0).values
        plt.bar(x + (i - len(show_cols) / 2) * width + width / 2, values, width, label=metric.upper())

    plt.xticks(x, models, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("指标值")
    plt.title("不同聚类模型综合指标对比")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    save_fig(output_path)


def plot_acc_bar(metrics_df: pd.DataFrame, output_path: Path):
    models = metrics_df["model"].astype(str).tolist()
    acc_values = metrics_df["acc"].astype(float).values

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, acc_values)

    for bar, value in zip(bars, acc_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylim(0, 1.05)
    plt.ylabel("ACC")
    plt.title("不同聚类模型 ACC 对比")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    save_fig(output_path)


def plot_runtime_bar(metrics_df: pd.DataFrame, output_path: Path):
    if "elapsed_seconds" not in metrics_df.columns:
        print("[WARN] 指标表没有 elapsed_seconds，跳过运行时间图。")
        return

    models = metrics_df["model"].astype(str).tolist()
    values = metrics_df["elapsed_seconds"].astype(float).fillna(0).values

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, values)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylabel("运行时间 / 秒")
    plt.title("不同聚类模型运行时间对比")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    save_fig(output_path)


def plot_crosstab_heatmap(result_df: pd.DataFrame, label_col: str, cluster_col: str, title: str, output_path: Path):
    ctab = pd.crosstab(result_df[cluster_col], result_df[label_col])

    plt.figure(figsize=(7, 5))
    plt.imshow(ctab.values, aspect="auto")
    plt.colorbar(label="样本数量")

    plt.xticks(np.arange(len(ctab.columns)), ctab.columns, rotation=20)
    plt.yticks(np.arange(len(ctab.index)), [f"cluster {x}" for x in ctab.index])

    for i in range(ctab.shape[0]):
        for j in range(ctab.shape[1]):
            plt.text(j, i, str(ctab.values[i, j]), ha="center", va="center", fontsize=10)

    plt.xlabel("真实类别")
    plt.ylabel("聚类簇")
    plt.title(title)

    save_fig(output_path)


def compute_2d_coordinates(
    df: pd.DataFrame,
    text_col: str,
    feature_mode: str,
    max_df: float,
    min_df: int,
    ngram_min: int,
    ngram_max: int,
    max_features: int | None,
    sublinear_tf: bool,
    seed: int,
):
    vectorizer = build_tfidf_vectorizer(
        feature_mode=feature_mode,
        max_df=max_df,
        min_df=min_df,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )

    X = vectorizer.fit_transform(df[text_col].fillna("").astype(str))
    reducer = make_pipeline(
        TruncatedSVD(n_components=2, random_state=seed),
        Normalizer(copy=False),
    )
    coords = reducer.fit_transform(X)
    return coords


def plot_scatter_by_label(coords, labels, title: str, output_path: Path):
    labels = pd.Series(labels).astype(str)
    unique_labels = sorted(labels.unique().tolist())

    plt.figure(figsize=(7, 5.5))

    for label in unique_labels:
        idx = labels == label
        plt.scatter(coords[idx, 0], coords[idx, 1], s=22, alpha=0.75, label=label)

    plt.xlabel("SVD-1")
    plt.ylabel("SVD-2")
    plt.title(title)
    plt.legend(markerscale=1.2, fontsize=9)
    plt.grid(linestyle="--", alpha=0.3)

    save_fig(output_path)


def plot_scatter_by_cluster(coords, clusters, title: str, output_path: Path):
    clusters = pd.Series(clusters).astype(str)
    unique_clusters = sorted(clusters.unique().tolist())

    plt.figure(figsize=(7, 5.5))

    for cluster in unique_clusters:
        idx = clusters == cluster
        plt.scatter(coords[idx, 0], coords[idx, 1], s=22, alpha=0.75, label=f"cluster {cluster}")

    plt.xlabel("SVD-1")
    plt.ylabel("SVD-2")
    plt.title(title)
    plt.legend(markerscale=1.2, fontsize=9)
    plt.grid(linestyle="--", alpha=0.3)

    save_fig(output_path)


def plot_radar(metrics_df: pd.DataFrame, output_path: Path):
    """
    Plot radar chart with original metric values.

    This version does NOT apply min-max normalization.
    ACC, NMI and ARI are naturally in [0, 1].
    Silhouette is also shown with its original value, although it is usually smaller.
    """
    metric_cols = ["acc", "nmi", "ari", "silhouette_cosine"]
    metric_cols = [c for c in metric_cols if c in metrics_df.columns]

    if len(metric_cols) < 3:
        print("[WARN] 雷达图至少需要 3 个指标，跳过。")
        return

    values = metrics_df[metric_cols].astype(float).fillna(0)

    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for idx, row in values.iterrows():
        model = metrics_df.loc[idx, "model"]
        row_values = row.tolist()
        row_values += row_values[:1]

        ax.plot(angles, row_values, linewidth=2, label=str(model))
        ax.fill(angles, row_values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.upper() for c in metric_cols])

    # 原始指标大多在 0~1 之间，因此固定雷达图半径范围为 0~1
    ax.set_ylim(0, 1.0)

    # 设置径向刻度，便于读数
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

    ax.set_title("不同聚类模型综合表现雷达图", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12))

    save_fig(output_path)

def plot_minibatch_grid_heatmap(grid_df: pd.DataFrame, output_path: Path):
    required_cols = {"batch_size", "n_init", "max_iter", "acc"}
    if not required_cols.issubset(set(grid_df.columns)):
        print("[WARN] MiniBatch grid 文件列不完整，跳过参数搜索热力图。")
        return

    # Use best ACC over max_iter for each batch_size × n_init.
    pivot_df = (
        grid_df.groupby(["batch_size", "n_init"])["acc"]
        .max()
        .reset_index()
        .pivot(index="batch_size", columns="n_init", values="acc")
    )

    plt.figure(figsize=(7, 5))
    plt.imshow(pivot_df.values, aspect="auto")
    plt.colorbar(label="Best ACC")

    plt.xticks(np.arange(len(pivot_df.columns)), pivot_df.columns)
    plt.yticks(np.arange(len(pivot_df.index)), pivot_df.index)

    for i in range(pivot_df.shape[0]):
        for j in range(pivot_df.shape[1]):
            value = pivot_df.values[i, j]
            if not np.isnan(value):
                plt.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10)

    plt.xlabel("n_init")
    plt.ylabel("batch_size")
    plt.title("MiniBatchKMeans 参数搜索热力图")

    save_fig(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="800")
    parser.add_argument("--input", default=str(DATA_DIR / "news_for_cluster_800.csv"))
    parser.add_argument("--metrics", default="")
    parser.add_argument("--result", default="")
    parser.add_argument("--mbk-grid", default="")
    parser.add_argument("--output-dir", default=str(FIG_DIR))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-mode", choices=["word", "char"], default="char")
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--sublinear-tf", default="true")

    args = parser.parse_args()
    setup_chinese_font()

    tag = args.tag
    input_path = Path(args.input)
    metrics_path = Path(args.metrics) if args.metrics else DATA_DIR / f"cluster_model_comparison_{tag}.csv"
    result_path = Path(args.result) if args.result else DATA_DIR / f"cluster_result_compare_{tag}.csv"
    mbk_grid_path = Path(args.mbk_grid) if args.mbk_grid else DATA_DIR / f"minibatch_kmeans_grid_{tag}.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_features = None if args.max_features <= 0 else args.max_features
    sublinear_tf = parse_bool(args.sublinear_tf)

    print("===== 读取可视化输入文件 =====")
    raw_df = pd.read_csv(input_path, encoding="utf-8-sig")
    metrics_df = pd.read_csv(metrics_path, encoding="utf-8-sig")
    result_df = pd.read_csv(result_path, encoding="utf-8-sig")

    label_col = find_column(result_df.columns, ["label", "category", "class", "tag", "type"])
    if label_col is None:
        label_col = find_column(raw_df.columns, ["label", "category", "class", "tag", "type"])

    if args.feature_mode == "char":
        text_col = find_column(raw_df.columns, ["text", "content", "text_cut"])
    else:
        text_col = find_column(raw_df.columns, ["text_cut", "text", "content"])

    if label_col is None:
        raise ValueError("Cannot find label column.")
    if text_col is None:
        raise ValueError("Cannot find text column.")

    print("===== 绘制指标对比图 =====")
    plot_metrics_grouped_bar(
        metrics_df,
        output_dir / f"model_metrics_comparison_{tag}.png",
    )
    plot_acc_bar(
        metrics_df,
        output_dir / f"model_acc_comparison_{tag}.png",
    )
    plot_runtime_bar(
        metrics_df,
        output_dir / f"model_runtime_comparison_{tag}.png",
    )
    plot_radar(
        metrics_df,
        output_dir / f"model_radar_comparison_{tag}.png",
    )

    print("===== 绘制交叉表热力图 =====")
    cluster_cols = [
        ("cluster_kmeans", "KMeans"),
        ("cluster_minibatch_kmeans", "MiniBatchKMeans"),
        ("cluster_agglomerative", "AgglomerativeClustering"),
    ]

    for cluster_col, model_name in cluster_cols:
        if cluster_col in result_df.columns:
            plot_crosstab_heatmap(
                result_df=result_df,
                label_col=label_col,
                cluster_col=cluster_col,
                title=f"{model_name} 聚类结果与真实类别交叉热力图",
                output_path=output_dir / f"crosstab_heatmap_{model_name}_{tag}.png",
            )
        else:
            print(f"[WARN] 缺少列 {cluster_col}，跳过 {model_name} 热力图。")

    print("===== 计算二维 SVD 坐标 =====")
    raw_df[text_col] = raw_df[text_col].fillna("").astype(str)
    coords = compute_2d_coordinates(
        df=raw_df,
        text_col=text_col,
        feature_mode=args.feature_mode,
        max_df=args.max_df,
        min_df=args.min_df,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        seed=args.seed,
    )

    print("===== 绘制二维散点图 =====")
    true_labels = raw_df[label_col].values if label_col in raw_df.columns else result_df[label_col].values

    plot_scatter_by_label(
        coords=coords,
        labels=true_labels,
        title="真实类别的二维 SVD 分布",
        output_path=output_dir / f"true_label_scatter_{tag}.png",
    )

    for cluster_col, model_name in cluster_cols:
        if cluster_col in result_df.columns:
            plot_scatter_by_cluster(
                coords=coords,
                clusters=result_df[cluster_col].values,
                title=f"{model_name} 聚类结果二维可视化",
                output_path=output_dir / f"cluster_scatter_{model_name}_{tag}.png",
            )

    if mbk_grid_path.exists():
        print("===== 绘制 MiniBatchKMeans 参数搜索热力图 =====")
        grid_df = pd.read_csv(mbk_grid_path, encoding="utf-8-sig")
        plot_minibatch_grid_heatmap(
            grid_df,
            output_dir / f"minibatch_grid_heatmap_{tag}.png",
        )
    else:
        print(f"[WARN] 未找到 MiniBatchKMeans grid 文件: {mbk_grid_path}")

    print("===== 可视化完成 =====")
    print(f"图像输出目录: {output_dir}")


if __name__ == "__main__":
    main()