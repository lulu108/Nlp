# -*- coding: utf-8 -*-
"""
Generate Figure 5-3: cluster keywords and purity analysis.

Inputs:
    P2/data/cluster_top_keywords_kmeans_800.csv
    P2/data/cluster_crosstab_kmeans_800.csv

Output:
    P2/data/figures/fig5_3_cluster_keywords_purity.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "figures"


KEYWORD_INFO = {
    0: {
        "label": "教育",
        "keywords": ["考生", "考试", "招生", "学校", "考研"],
        "desc": "考试升学与招生信息",
    },
    1: {
        "label": "科技",
        "keywords": ["公司", "市场", "用户", "产品", "信息"],
        "desc": "公司、用户与产品信息",
    },
    2: {
        "label": "房产",
        "keywords": ["平米", "户型", "楼盘", "项目", "房产"],
        "desc": "户型面积与楼盘项目",
    },
    3: {
        "label": "体育",
        "keywords": ["比赛", "球队", "球员", "NBA", "赛季"],
        "desc": "赛事、球队与球员动态",
    },
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


def load_crosstab(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到交叉表文件: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", index_col=0)


def plot_keywords_purity(ctab: pd.DataFrame, output_path: Path) -> None:
    set_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, cluster_id in zip(axes, sorted(KEYWORD_INFO.keys())):
        info = KEYWORD_INFO[cluster_id]

        row = ctab.loc[cluster_id]
        total = int(row.sum())
        main_label = row.idxmax()
        main_count = int(row.max())
        purity = main_count / total if total > 0 else 0.0

        keywords = info["keywords"]
        desc = info["desc"]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        # 标题
        ax.text(
            0.5, 0.84,
            f"cluster {cluster_id}  →  {info['label']}",
            ha="center", va="center",
            fontsize=18, fontweight="bold"
        )

        # 样本数与主类别占比
        ax.text(
            0.5, 0.68,
            f"样本数：{total}    主类别占比：{main_label} {purity:.1%}",
            ha="center", va="center",
            fontsize=12
        )

        # 关键词分层展示
        ax.text(
            0.5, 0.49,
            " / ".join(keywords[:3]),
            ha="center", va="center",
            fontsize=20, fontweight="bold"
        )

        ax.text(
            0.5, 0.34,
            " / ".join(keywords[3:]),
            ha="center", va="center",
            fontsize=14
        )

        # 解释
        ax.text(
            0.5, 0.18,
            f"主题：{desc}",
            ha="center", va="center",
            fontsize=12
        )

    fig.suptitle("各聚类簇关键词与类别纯度分析", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crosstab",
        default=str(DATA_DIR / "cluster_crosstab_kmeans_800.csv"),
        help="cluster crosstab CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output figure directory",
    )
    args = parser.parse_args()

    ctab = load_crosstab(Path(args.crosstab))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "fig5_3_cluster_keywords_purity.png"
    plot_keywords_purity(ctab, output_path)

    print("===== 图5-3生成完成 =====")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()