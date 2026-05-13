# -*- coding: utf-8 -*-
"""
Generate a three-panel crosstab heatmap for clustering model comparison.

Inputs:
    P2/data/cluster_crosstab_compare_800_KMeans.csv
    P2/data/cluster_crosstab_compare_800_MiniBatchKMeans.csv
    P2/data/cluster_crosstab_compare_800_AgglomerativeClustering.csv

Output:
    P2/data/figures/fig6_10_model_crosstab_compare.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = DATA_DIR / "figures"


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_crosstab(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到交叉表文件: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    # 第一列是 cluster 编号列，例如 cluster_kmeans / cluster_minibatch_kmeans / cluster_agglomerative
    cluster_col = df.columns[0]
    df = df.set_index(cluster_col)

    # 统一行名，防止不同文件第一列名称不同导致图中显示不一致
    df.index = [f"cluster {i}" for i in df.index]

    # 保证类别列顺序一致
    label_order = ["体育", "房产", "教育", "科技"]
    existing_cols = [c for c in label_order if c in df.columns]
    df = df[existing_cols]

    return df


def add_heatmap(ax, data: pd.DataFrame, title: str, vmax: float) -> None:
    im = ax.imshow(data.values, aspect="auto", vmin=0, vmax=vmax)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)
    ax.set_xlabel("真实类别")
    ax.set_ylabel("聚类簇")

    # 写入每个格子的数字
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = int(data.iloc[i, j])
            # 根据背景深浅调整文字颜色
            text_color = "white" if value > vmax * 0.55 else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    return im


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="800")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(FIG_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    set_chinese_font()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "KMeans": data_dir / f"cluster_crosstab_compare_{args.tag}_KMeans.csv",
        "MiniBatchKMeans": data_dir / f"cluster_crosstab_compare_{args.tag}_MiniBatchKMeans.csv",
        "AgglomerativeClustering": data_dir / f"cluster_crosstab_compare_{args.tag}_AgglomerativeClustering.csv",
    }

    crosstabs = {name: read_crosstab(path) for name, path in paths.items()}

    vmax = max(float(df.values.max()) for df in crosstabs.values())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)

    last_im = None
    for ax, (name, df) in zip(axes, crosstabs.items()):
        last_im = add_heatmap(ax, df, name, vmax=vmax)

    fig.suptitle("不同聚类模型的 cluster 与真实类别交叉热力图对比", fontsize=18)

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, location="right")
    cbar.set_label("样本数量", fontsize=12)

    output_path = output_dir / f"fig6_10_model_crosstab_compare_{args.tag}.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("===== 三联交叉热力图生成完成 =====")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()