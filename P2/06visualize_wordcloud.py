# -*- coding: utf-8 -*-
"""
Generate word clouds from cluster_result_kmeans_800.csv.

Logic:
1. Read cluster_result_kmeans_800.csv.
2. Group samples by cluster.
3. Count words in text_cut/text column.
4. Remove single-character words, digits and generic words.
5. Keep top-N words for each cluster.
6. Generate one 2x2 word cloud figure.

Input:
    P2/data/cluster_result_kmeans_800.csv

Output:
    P2/data/figures/fig5_4_cluster_wordcloud_from_result.png
    P2/data/figures/cluster_word_freq_800.csv
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "cluster_result_kmeans_800.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "figures"


# 泛化词，可根据你的结果继续补充
GENERIC_STOPWORDS = {
    "一个", "一种", "一些", "这个", "这些", "那个", "那些",
    "可以", "可能", "进行", "表示", "认为", "目前", "已经", "没有",
    "记者", "报道", "消息", "时间", "北京", "中国", "今年", "去年",
    "开始", "出现", "自己", "相关", "方面", "通过", "由于", "因为",
    "以及", "其中", "对于", "如果", "但是", "还是", "成为", "需要",
    "信息", "市场", "公司",  # 这几个是否删除要看你希望词云保留多少泛化词
}


LABEL_HINT = {
    0: "教育",
    1: "科技",
    2: "房产",
    3: "体育",
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


def find_text_column(df: pd.DataFrame) -> str:
    """
    Prefer text_cut. Fall back to common text columns.
    """
    candidates = ["text_cut", "text", "content"]
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]

    for c in df.columns:
        c_lower = c.lower()
        if "text" in c_lower or "content" in c_lower:
            return c

    raise ValueError(f"未找到文本列，当前列名为: {list(df.columns)}")


def is_valid_word(word: str, min_len: int) -> bool:
    word = word.strip()
    if not word:
        return False
    if len(word) < min_len:
        return False
    if word in GENERIC_STOPWORDS:
        return False
    if re.fullmatch(r"\d+(\.\d+)?", word):
        return False
    if re.fullmatch(r"[a-zA-Z]+", word) and len(word) <= 1:
        return False
    if re.fullmatch(r"[\W_]+", word):
        return False
    return True


def count_words(texts: pd.Series, min_len: int, top_n: int) -> Counter:
    counter = Counter()

    for text in texts.fillna("").astype(str):
        # text_cut 通常已经是空格分隔
        tokens = text.split()
        for tok in tokens:
            tok = tok.strip()
            if is_valid_word(tok, min_len=min_len):
                counter[tok] += 1

    return Counter(dict(counter.most_common(top_n)))


def get_cluster_label(df_cluster: pd.DataFrame, cluster_id: int) -> str:
    if "mapped_label_name" in df_cluster.columns:
        value_counts = df_cluster["mapped_label_name"].value_counts()
        if len(value_counts) > 0:
            return str(value_counts.index[0])
    return LABEL_HINT.get(cluster_id, "未知")


def generate_wordcloud_figure(
    cluster_freqs: dict[int, Counter],
    cluster_labels: dict[int, str],
    output_path: Path,
    font_path: str,
) -> None:
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise ImportError("未安装 wordcloud，请先运行: pip install wordcloud") from exc

    set_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    axes = axes.flatten()

    for ax, cluster_id in zip(axes, sorted(cluster_freqs.keys())):
        freqs = cluster_freqs[cluster_id]
        label = cluster_labels.get(cluster_id, "未知")

        if not freqs:
            ax.text(0.5, 0.5, "无有效关键词", ha="center", va="center")
            ax.axis("off")
            continue

        wc = WordCloud(
            font_path=font_path,
            width=900,
            height=500,
            background_color="white",
            max_words=80,
            collocations=False,
            prefer_horizontal=0.9,
            random_state=42,
        )
        wc.generate_from_frequencies(freqs)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"cluster {cluster_id} → {label}", fontsize=16, fontweight="bold")
        ax.axis("off")

    fig.suptitle("各聚类簇关键词词云图", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_freq_table(
    cluster_freqs: dict[int, Counter],
    cluster_labels: dict[int, str],
    output_path: Path,
) -> None:
    rows = []
    for cluster_id, freqs in cluster_freqs.items():
        for word, freq in freqs.items():
            rows.append(
                {
                    "cluster": cluster_id,
                    "mapped_label_name": cluster_labels.get(cluster_id, "未知"),
                    "word": word,
                    "freq": freq,
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="cluster result csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output figure directory")
    parser.add_argument("--font-path", default="C:/Windows/Fonts/msyh.ttc", help="Chinese font path")
    parser.add_argument("--top-n", type=int, default=20, help="top words per cluster")
    parser.add_argument("--min-len", type=int, default=2, help="minimum word length")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"未找到聚类结果文件: {input_path}")

    font_path = Path(args.font_path)
    if not font_path.exists():
        raise FileNotFoundError(
            f"未找到中文字体文件: {font_path}\n"
            f"Windows可尝试: C:/Windows/Fonts/msyh.ttc 或 C:/Windows/Fonts/simhei.ttf"
        )

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if "cluster" not in df.columns:
        raise ValueError(f"未找到 cluster 列，当前列名为: {list(df.columns)}")

    text_col = find_text_column(df)

    cluster_freqs: dict[int, Counter] = {}
    cluster_labels: dict[int, str] = {}

    for cluster_id, df_cluster in df.groupby("cluster"):
        cluster_id = int(cluster_id)
        label = get_cluster_label(df_cluster, cluster_id)
        freqs = count_words(df_cluster[text_col], min_len=args.min_len, top_n=args.top_n)

        cluster_freqs[cluster_id] = freqs
        cluster_labels[cluster_id] = label

    fig_path = output_dir / "fig5_4_cluster_wordcloud_from_result.png"
    csv_path = output_dir / "cluster_word_freq_800.csv"

    generate_wordcloud_figure(
        cluster_freqs=cluster_freqs,
        cluster_labels=cluster_labels,
        output_path=fig_path,
        font_path=str(font_path),
    )
    save_freq_table(cluster_freqs, cluster_labels, csv_path)

    print("===== 词云图生成完成 =====")
    print(f"使用文本列: {text_col}")
    print(f"词云图: {fig_path}")
    print(f"词频表: {csv_path}")

    for cluster_id in sorted(cluster_freqs.keys()):
        print(f"\ncluster {cluster_id} -> {cluster_labels[cluster_id]}")
        print(cluster_freqs[cluster_id].most_common(10))


if __name__ == "__main__":
    main()