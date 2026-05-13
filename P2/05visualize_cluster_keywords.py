# -*- coding: utf-8 -*-
"""
Visualize Top keywords of KMeans clusters for Experiment 2.

Input:
    P2/data/cluster_top_keywords_kmeans_800.csv

Output:
    P2/data/figures/fig5_3_cluster_keywords_cards.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "figures"


# 这里手动整理可读性更强的关键词。
# 原始CSV是char n-gram特征，会出现一些单字和重复片段。
READABLE_KEYWORDS = {
    0: ["考生", "考试", "招生", "学校", "考研"],
    1: ["公司", "市场", "用户", "产品", "信息"],
    2: ["平米", "户型", "楼盘", "项目", "房产"],
    3: ["比赛", "球队", "球员", "NBA", "赛季"],
}


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_keyword_cards(keyword_df: pd.DataFrame, output_path: Path) -> None:
    set_chinese_font()

    keyword_df = keyword_df.sort_values("cluster").copy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, keyword_df.iterrows()):
        cluster_id = int(row["cluster"])
        label_name = str(row["mapped_label_name"])
        keywords = READABLE_KEYWORDS.get(cluster_id)

        if not keywords:
            raw_keywords = str(row["top_keywords"]).split()
            keywords = raw_keywords[:8]

        title = f"cluster {cluster_id} → {label_name}"
        text = " / ".join(keywords)

        ax.text(
            0.5,
            0.62,
            title,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.38,
            text,
            ha="center",
            va="center",
            fontsize=13,
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)

    fig.suptitle("各聚类簇代表性关键词", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "cluster_top_keywords_kmeans_800.csv"),
        help="cluster top keywords csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output figure directory",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"未找到关键词文件: {input_path}")

    keyword_df = pd.read_csv(input_path, encoding="utf-8-sig")
    output_path = output_dir / "fig5_3_cluster_keywords_cards.png"

    plot_keyword_cards(keyword_df=keyword_df, output_path=output_path)

    print("===== 各簇关键词图生成完成 =====")
    print(f"图5-3: {output_path}")


if __name__ == "__main__":
    main()