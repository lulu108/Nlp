#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
阶段 6：实验三核心可视化图片生成

本脚本只读取已有数据和实验结果，不重新训练模型。
运行方式：
    python P3/06_visualize_results.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "visualization_outputs"

CLASS_DATA_PATH = DATA_DIR / "thucnews_4class_200_clean.csv"
MODEL_SUMMARY_PATH = DATA_DIR / "classical_model_outputs" / "classical_models_summary.csv"
SVM_CM_SOURCE_PATH = DATA_DIR / "classical_model_outputs" / "svm_test_confusion_matrix.png"
MULTILABEL_REPORT_PATH = DATA_DIR / "multilabel_outputs" / "multilabel_report.txt"

CLASS_DISTRIBUTION_OUT = OUTPUT_DIR / "class_distribution.png"
MODEL_COMPARISON_OUT = OUTPUT_DIR / "model_comparison.png"
SVM_CM_OUT = OUTPUT_DIR / "svm_confusion_matrix.png"
MULTILABEL_F1_OUT = OUTPUT_DIR / "multilabel_label_f1.png"

LABELS = ["体育", "科技", "财经", "教育"]
MODEL_ORDER = ["NB", "LR", "SVM"]


def setup_matplotlib() -> None:
    """设置中文字体和负号显示。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> pd.DataFrame:
    """读取 UTF-8 CSV，兼容带 BOM 的文件。"""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def add_bar_labels(ax, bars, fmt: str = "{:.0f}") -> None:
    """在柱子上方标注数值。"""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_class_distribution() -> Path | None:
    """生成图 3-1：四类文本样本数量分布。"""
    if not CLASS_DATA_PATH.exists():
        print(f"[跳过] 未找到类别分布数据文件：{CLASS_DATA_PATH}")
        return None

    df = read_csv(CLASS_DATA_PATH)
    if "category" not in df.columns:
        print(f"[跳过] 数据文件缺少 category 列：{CLASS_DATA_PATH}")
        return None

    counts = df["category"].value_counts().reindex(LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color="#4C78A8")
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数量")
    ax.set_title("四类文本样本数量分布")
    ax.set_ylim(0, max(counts.values) * 1.15 if len(counts) else 1)
    add_bar_labels(ax, bars, "{:.0f}")
    fig.tight_layout()
    fig.savefig(CLASS_DISTRIBUTION_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return CLASS_DISTRIBUTION_OUT


def plot_model_comparison() -> Path | None:
    """生成图 3-2：不同分类模型测试集性能对比。"""
    if not MODEL_SUMMARY_PATH.exists():
        print(f"[跳过] 未找到模型汇总文件：{MODEL_SUMMARY_PATH}")
        return None

    df = read_csv(MODEL_SUMMARY_PATH)
    required_columns = {"model", "test_acc", "test_macro_f1"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"[跳过] 模型汇总文件缺少字段：{sorted(missing_columns)}")
        return None

    df = df.set_index("model").reindex(MODEL_ORDER)
    if df[["test_acc", "test_macro_f1"]].isna().any().any():
        print(f"[提示] 模型汇总中存在缺失模型或指标，将按现有数据绘制：{MODEL_SUMMARY_PATH}")
    df = df.reset_index()

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    acc_bars = ax.bar(x - width / 2, df["test_acc"], width, label="Accuracy", color="#4C78A8")
    f1_bars = ax.bar(x + width / 2, df["test_macro_f1"], width, label="Macro-F1", color="#F58518")
    ax.set_xlabel("模型")
    ax.set_ylabel("指标值")
    ax.set_title("不同分类模型测试集性能对比")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 1.08)
    ax.legend()
    add_bar_labels(ax, acc_bars, "{:.4f}")
    add_bar_labels(ax, f1_bars, "{:.4f}")
    fig.tight_layout()
    fig.savefig(MODEL_COMPARISON_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return MODEL_COMPARISON_OUT


def copy_svm_confusion_matrix() -> Path | None:
    """生成图 3-3：复制 SVM 混淆矩阵图片。"""
    if not SVM_CM_SOURCE_PATH.exists():
        print(f"[跳过] 未找到 SVM 混淆矩阵图片：{SVM_CM_SOURCE_PATH}")
        return None

    shutil.copy2(SVM_CM_SOURCE_PATH, SVM_CM_OUT)
    return SVM_CM_OUT


def parse_multilabel_f1_scores(report_text: str) -> dict[str, float]:
    """从 classification_report 文本中解析四个标签的 f1-score。"""
    f1_scores: dict[str, float] = {}
    for label in LABELS:
        pattern = rf"^\s*{re.escape(label)}\s+[\d.]+\s+[\d.]+\s+([\d.]+)"
        match = re.search(pattern, report_text, flags=re.MULTILINE)
        if match:
            f1_scores[label] = float(match.group(1))
    return f1_scores


def plot_multilabel_label_f1() -> Path | None:
    """生成图 3-4：多标签分类各标签 F1-score 对比。"""
    if not MULTILABEL_REPORT_PATH.exists():
        print(f"[跳过] 未找到多标签分类报告：{MULTILABEL_REPORT_PATH}")
        return None

    report_text = MULTILABEL_REPORT_PATH.read_text(encoding="utf-8-sig")
    f1_scores = parse_multilabel_f1_scores(report_text)
    missing_labels = [label for label in LABELS if label not in f1_scores]
    if missing_labels:
        print(f"[跳过] 多标签报告中未解析到以下标签 F1-score：{missing_labels}")
        return None

    values = [f1_scores[label] for label in LABELS]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(LABELS, values, color="#54A24B")
    ax.set_xlabel("标签")
    ax.set_ylabel("F1-score")
    ax.set_title("多标签分类各标签 F1-score 对比")
    ax.set_ylim(0, 1.08)
    add_bar_labels(ax, bars, "{:.4f}")
    fig.tight_layout()
    fig.savefig(MULTILABEL_F1_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return MULTILABEL_F1_OUT


def main() -> None:
    setup_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_paths = []
    for path in (
        plot_class_distribution(),
        plot_model_comparison(),
        copy_svm_confusion_matrix(),
        plot_multilabel_label_f1(),
    ):
        if path is not None:
            generated_paths.append(path)

    if generated_paths:
        print("\n成功生成或复制的图片：")
        for path in generated_paths:
            print(path)
    else:
        print("\n未生成任何图片，请检查输入文件是否存在。")


if __name__ == "__main__":
    main()
