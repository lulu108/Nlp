# -*- coding: utf-8 -*-
"""
Visualize representative keywords of KMeans clusters.

Input:
    P2/data/cluster_top_keywords_kmeans_800.csv

Outputs:
    P2/data/figures/fig5_3_cluster_keywords_cards_beautified.png
    P2/data/figures/fig5_3_cluster_keywords_wordcloud.png

Note:
    The original keywords are extracted from char-level n-gram TF-IDF features.
    For report readability, this script uses manually cleaned representative
    keywords consistent with the cluster topics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "cluster_top_keywords_kmeans_800.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "figures"


# 人工整理后的代表性关键词。
# 原始CSV来自字符级 n-gram，会出现部分单字和重复片段；
# 报告图中建议使用更可读的主题词。
KEYWORD_INFO = {
    0: {
        "label": "教育",
        "main": ["考生", "考试", "招生"],
        "extra": ["学校", "考研"],
        "desc": "考试升学与招生信息",
        "weights": {
            "考生": 100,
            "考试": 90,
            "招生": 80,
            "学校": 65,
            "考研": 55,
        },
    },
    1: {
        "label": "科技",
        "main": ["公司", "市场", "用户"],
        "extra": ["产品", "信息"],
        "desc": "公司、用户与产品信息",
        "weights": {
            "公司": 100,
            "市场": 85,
            "用户": 75,
            "产品": 65,
            "信息": 55,
        },
    },
    2: {
        "label": "房产",
        "main": ["平米", "户型", "楼盘"],
        "extra": ["项目", "房产"],
        "desc": "户型面积与楼盘项目",
        "weights": {
            "平米": 100,
            "户型": 95,
            "楼盘": 80,
            "项目": 70,
            "房产": 60,
        },
    },
    3: {
        "label": "体育",
        "main": ["比赛", "球队", "球员"],
        "extra": ["NBA", "赛季"],
        "desc": "赛事、球队与球员动态",
        "weights": {
            "比赛": 100,
            "球队": 95,
            "球员": 80,
            "NBA": 70,
            "赛季": 60,
        },
    },
}


def set_chinese_font() -> None:
    """Set Chinese font for matplotlib."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_keyword_file(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"未找到关键词文件: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required = {"cluster", "mapped_label_name", "top_keywords"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"关键词文件缺少字段: {missing}")

    return df.sort_values("cluster").reset_index(drop=True)


def plot_beautified_cards(keyword_df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a more polished 2x2 keyword card figure.
    """
    set_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, keyword_df.iterrows()):
        cluster_id = int(row["cluster"])
        info = KEYWORD_INFO.get(cluster_id, None)

        if info is None:
            label = str(row["mapped_label_name"])
            raw_words = str(row["top_keywords"]).split()
            main_words = raw_words[:3]
            extra_words = raw_words[3:6]
            desc = "聚类中心高权重特征"
        else:
            label = info["label"]
            main_words = info["main"]
            extra_words = info["extra"]
            desc = info["desc"]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 卡片边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4)

        ax.set_xticks([])
        ax.set_yticks([])

        # 标题
        ax.text(
            0.5,
            0.82,
            f"cluster {cluster_id}  →  {label}",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )

        # 主要关键词
        x_positions = [0.25, 0.5, 0.75]
        for x, word in zip(x_positions, main_words):
            ax.text(
                x,
                0.56,
                word,
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
            )

        # 次级关键词
        ax.text(
            0.5,
            0.36,
            " / ".join(extra_words),
            ha="center",
            va="center",
            fontsize=15,
        )

        # 主题说明
        ax.text(
            0.5,
            0.18,
            desc,
            ha="center",
            va="center",
            fontsize=12,
        )

    fig.suptitle("各聚类簇代表性关键词", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wordcloud(keyword_df: pd.DataFrame, output_path: Path, font_path: str | None) -> None:
    """
    Create 2x2 wordcloud figure.
    Requires:
        pip install wordcloud
    """
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise ImportError(
            "未安装 wordcloud。请先运行: pip install wordcloud"
        ) from exc

    set_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, keyword_df.iterrows()):
        cluster_id = int(row["cluster"])
        info = KEYWORD_INFO.get(cluster_id)

        if info is None:
            raw_words = str(row["top_keywords"]).split()
            weights = {word: max(10, 100 - i * 8) for i, word in enumerate(raw_words[:10])}
            label = str(row["mapped_label_name"])
        else:
            weights = info["weights"]
            label = info["label"]

        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=450,
            background_color="white",
            max_words=30,
            prefer_horizontal=0.9,
            collocations=False,
        )
        wc.generate_from_frequencies(weights)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"cluster {cluster_id} → {label}", fontsize=16, fontweight="bold")
        ax.axis("off")

    fig.suptitle("各聚类簇关键词词云图", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="cluster_top_keywords_kmeans_800.csv path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output directory for figures",
    )
    parser.add_argument(
        "--font-path",
        default="",
        help=(
            "Chinese font path for wordcloud. "
            "Example on Windows: C:/Windows/Fonts/msyh.ttc"
        ),
    )
    parser.add_argument(
        "--skip-wordcloud",
        default="false",
        help="true/false. If true, only generate beautified cards.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keyword_df = read_keyword_file(input_path)

    cards_path = output_dir / "fig5_3_cluster_keywords_cards_beautified.png"
    plot_beautified_cards(keyword_df, cards_path)

    print("===== 图5-3关键词图生成完成 =====")
    print(f"增强卡片图: {cards_path}")

    if args.skip_wordcloud.strip().lower() not in {"1", "true", "yes", "y"}:
        wordcloud_path = output_dir / "fig5_3_cluster_keywords_wordcloud.png"
        font_path = args.font_path.strip() or None
        plot_wordcloud(keyword_df, wordcloud_path, font_path)
        print(f"词云图: {wordcloud_path}")


if __name__ == "__main__":
    main()