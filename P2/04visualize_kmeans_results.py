# -*- coding: utf-8 -*-
"""
Visualize K-means clustering results for Experiment 2.

Generated figures:
1. fig5_1_kmeans_metrics_bar.png
   K-means metrics comparison for 440/600/800 datasets.

2. fig5_2_cluster_crosstab_heatmap.png
   Heatmap of cluster × true label distribution for 800 dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "figures"


def set_chinese_font() -> None:
    """
    Try to set a Chinese font for matplotlib.
    This avoids garbled Chinese characters in figure titles and labels.
    """
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_metrics(data_dir: Path, tags: list[str]) -> pd.DataFrame:
    """
    Read KMeans metrics files for different dataset sizes.
    """
    rows = []

    for tag in tags:
        file_path = data_dir / f"cluster_metrics_kmeans_{tag}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"未找到指标文件: {file_path}")

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if df.empty:
            raise ValueError(f"指标文件为空: {file_path}")

        row = df.iloc[0].to_dict()
        rows.append(
            {
                "数据组": f"{tag}组",
                "ACC": float(row["acc"]),
                "NMI": float(row["nmi"]),
                "ARI": float(row["ari"]),
                "Silhouette": float(row["silhouette_cosine"]),
            }
        )

    return pd.DataFrame(rows)


def plot_metrics_bar(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot grouped bar chart for ACC/NMI/ARI/Silhouette across datasets.
    """
    metric_names = ["ACC", "NMI", "ARI", "Silhouette"]
    groups = metrics_df["数据组"].tolist()

    x = np.arange(len(metric_names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, group in enumerate(groups):
        values = metrics_df.loc[metrics_df["数据组"] == group, metric_names].iloc[0].values
        positions = x + (i - (len(groups) - 1) / 2) * width
        bars = ax.bar(positions, values, width, label=group)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("K-means 聚类指标对比")
    ax.set_ylabel("指标值")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_crosstab(data_dir: Path, tag: str) -> pd.DataFrame:
    """
    Read cluster × true label crosstab file.
    """
    file_path = data_dir / f"cluster_crosstab_kmeans_{tag}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"未找到交叉表文件: {file_path}")

    ctab = pd.read_csv(file_path, encoding="utf-8-sig", index_col=0)
    return ctab


def plot_crosstab_heatmap(ctab: pd.DataFrame, output_path: Path) -> None:
    """
    Plot heatmap for cluster × true label distribution.
    """
    values = ctab.values
    row_labels = [f"cluster {idx}" for idx in ctab.index]
    col_labels = ctab.columns.tolist()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(values)

    ax.set_title("cluster 与真实类别对应关系热力图")
    ax.set_xlabel("真实类别")
    ax.set_ylabel("聚类簇")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    max_value = values.max() if values.size else 1

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = int(values[i, j])
            text_color = "white" if value > max_value * 0.55 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="P2 data directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for generated figures",
    )
    parser.add_argument(
        "--metric-tags",
        default="440,600,800",
        help="dataset tags for metrics bar chart, e.g. 440,600,800",
    )
    parser.add_argument(
        "--crosstab-tag",
        default="800",
        help="dataset tag for cluster crosstab heatmap",
    )
    args = parser.parse_args()

    set_chinese_font()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tags = [tag.strip() for tag in args.metric_tags.split(",") if tag.strip()]

    metrics_df = read_metrics(data_dir=data_dir, tags=tags)
    metrics_output = output_dir / "fig5_1_kmeans_metrics_bar.png"
    plot_metrics_bar(metrics_df=metrics_df, output_path=metrics_output)

    ctab = read_crosstab(data_dir=data_dir, tag=args.crosstab_tag)
    heatmap_output = output_dir / "fig5_2_cluster_crosstab_heatmap.png"
    plot_crosstab_heatmap(ctab=ctab, output_path=heatmap_output)

    print("===== 可视化图生成完成 =====")
    print(f"图5-1: {metrics_output}")
    print(f"图5-2: {heatmap_output}")
    print("\n表5-1 数据:")
    print(metrics_df.to_string(index=False))
    print("\ncluster × 真实类别交叉表:")
    print(ctab)


if __name__ == "__main__":
    main()