# -*- coding: utf-8 -*-
"""
Visualize KMeans baseline vs optimized setting.

Inputs:
    P2/data/cluster_metrics_kmeans_800.csv
    P2/data/cluster_metrics_kmeans_800_kmeans_opt.csv

Outputs:
    P2/data/figures/fig6_2_kmeans_baseline_opt_bar.png
    P2/data/figures/fig6_3_kmeans_baseline_opt_radar.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "figures"


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_metric_row(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"未找到指标文件: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"指标文件为空: {path}")

    row = df.iloc[0].to_dict()

    return {
        "ACC": float(row["acc"]),
        "NMI": float(row["nmi"]),
        "ARI": float(row["ari"]),
        "Silhouette": float(row["silhouette_cosine"])
        if not pd.isna(row["silhouette_cosine"])
        else np.nan,
    }


def plot_bar(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metric_names = ["ACC", "NMI", "ARI", "Silhouette"]
    x = np.arange(len(metric_names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))

    baseline_values = metrics_df.loc["基线方案", metric_names].values.astype(float)
    opt_values = metrics_df.loc["优化方案", metric_names].values.astype(float)

    bars1 = ax.bar(x - width / 2, baseline_values, width, label="基线方案")
    bars2 = ax.bar(x + width / 2, opt_values, width, label="优化方案")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if math.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("K-means 基线方案与优化方案指标对比")
    ax.set_ylabel("指标值")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_radar(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metric_names = ["ACC", "NMI", "ARI", "Silhouette"]

    baseline_values = metrics_df.loc["基线方案", metric_names].values.astype(float)
    opt_values = metrics_df.loc["优化方案", metric_names].values.astype(float)

    if np.isnan(baseline_values).any() or np.isnan(opt_values).any():
        print("[WARN] 雷达图存在 NaN 指标，将跳过 NaN 指标。")
        valid_mask = ~np.isnan(baseline_values) & ~np.isnan(opt_values)
        metric_names = [m for m, keep in zip(metric_names, valid_mask) if keep]
        baseline_values = baseline_values[valid_mask]
        opt_values = opt_values[valid_mask]

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    baseline = baseline_values.tolist()
    opt = opt_values.tolist()

    angles += angles[:1]
    baseline += baseline[:1]
    opt += opt[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})

    ax.plot(angles, baseline, linewidth=2, label="基线方案")
    ax.fill(angles, baseline, alpha=0.12)

    ax.plot(angles, opt, linewidth=2, label="优化方案")
    ax.fill(angles, opt, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.set_title("K-means 基线方案与优化方案雷达图", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default=str(DATA_DIR / "cluster_metrics_kmeans_800.csv"),
        help="baseline metrics csv",
    )
    parser.add_argument(
        "--optimized",
        default=str(DATA_DIR / "cluster_metrics_kmeans_800_kmeans_opt.csv"),
        help="optimized metrics csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output figure directory",
    )
    args = parser.parse_args()

    set_chinese_font()

    baseline_path = Path(args.baseline)
    optimized_path = Path(args.optimized)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = read_metric_row(baseline_path)
    optimized_metrics = read_metric_row(optimized_path)

    metrics_df = pd.DataFrame(
        [baseline_metrics, optimized_metrics],
        index=["基线方案", "优化方案"],
    )

    print("===== K-means 基线/优化指标 =====")
    print(metrics_df)

    bar_path = output_dir / "fig6_2_kmeans_baseline_opt_bar.png"
    radar_path = output_dir / "fig6_3_kmeans_baseline_opt_radar.png"

    plot_bar(metrics_df, bar_path)
    plot_radar(metrics_df, radar_path)

    print("===== 图片生成完成 =====")
    print(f"柱状图: {bar_path}")
    print(f"雷达图: {radar_path}")


if __name__ == "__main__":
    main()