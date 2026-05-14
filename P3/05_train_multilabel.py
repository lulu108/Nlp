#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
阶段 5：中文多标签文本分类扩展实验

功能：
1. 读取 P3/data/multilabel_news_200.csv
2. 使用 jieba 对 text 分词，并加载停用词表
3. 构造 text_processed 作为模型输入
4. 按 8:2 划分训练集和测试集
5. 仅在训练集上 fit TfidfVectorizer，避免数据泄漏
6. 使用 OneVsRestClassifier(LogisticRegression(max_iter=2000)) 训练模型
7. 输出多标签分类指标与 classification_report
8. 保存报告、预测结果、模型和向量器
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jieba
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INPUT_CSV = DATA_DIR / "multilabel_news_200.csv"
DEFAULT_STOPWORDS = DATA_DIR / "stopwords.txt"
DEFAULT_OUTPUT_DIR = DATA_DIR / "multilabel_outputs"

LABEL_COLUMNS = ["体育", "科技", "财经", "教育"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="中文多标签文本分类扩展实验")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="多标签数据集路径")
    parser.add_argument("--stopwords", default=str(DEFAULT_STOPWORDS), help="停用词表路径")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="结果输出目录")
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    """读取 CSV，优先使用 utf-8-sig，兼容常见 UTF-8 文件。"""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def load_stopwords(path: Path) -> set[str]:
    """加载停用词表；如果文件不存在，则返回空集合。"""
    if not path.exists():
        print(f"[提示] 未找到停用词表：{path}，将使用空停用词表。")
        return set()

    for encoding in ("utf-8-sig", "utf-8"):
        try:
            words = path.read_text(encoding=encoding).splitlines()
            return {word.strip() for word in words if word.strip()}
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"无法读取停用词文件：{path}",
    )


def preprocess_text(text: object, stopwords: set[str]) -> str:
    """使用 jieba 分词并过滤空白词与停用词。"""
    text = "" if pd.isna(text) else str(text)
    tokens = []
    for token in jieba.lcut(text):
        token = token.strip()
        if not token:
            continue
        if token in stopwords:
            continue
        tokens.append(token)
    return " ".join(tokens)


def labels_to_string(label_values: pd.Series | list[int]) -> str:
    """将 0/1 标签列转换为分号分隔的中文标签字符串。"""
    result = ";".join(
        label_name
        for label_name, value in zip(LABEL_COLUMNS, label_values)
        if int(value) == 1
    )
    return result if result else "无"


def build_data_stats(
    df: pd.DataFrame,
    y: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """汇总数据集规模与标签分布信息。"""
    label_count_per_sample = y.sum(axis=1)
    return {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "label_counts": y.sum(axis=0).astype(int).to_dict(),
        "single_label_samples": int((label_count_per_sample == 1).sum()),
        "double_label_samples": int((label_count_per_sample == 2).sum()),
        "triple_label_samples": int((label_count_per_sample == 3).sum()),
        "quad_label_samples": int((label_count_per_sample == 4).sum()),
        "avg_labels_per_sample": float(label_count_per_sample.mean()),
    }


def save_report(
    report_path: Path,
    metrics: dict[str, float],
    data_stats: dict,
    cls_report: str,
    input_csv: Path,
    stopwords_path: Path,
) -> None:
    """保存指标汇总与分类报告。"""
    lines = [
        "中文多标签文本分类实验报告",
        "",
        f"输入数据: {input_csv}",
        f"停用词表: {stopwords_path if stopwords_path.exists() else '未提供，使用空停用词表'}",
        "",
        "数据统计信息:",
        "任务类型: 多标签文本分类",
        f"样本总数: {data_stats['total_samples']}",
        f"训练集样本数: {data_stats['train_samples']}",
        f"测试集样本数: {data_stats['test_samples']}",
        f"标签集合: {', '.join(LABEL_COLUMNS)}",
        "各标签出现次数:",
        *[
            f"{label}: {data_stats['label_counts'][label]}"
            for label in LABEL_COLUMNS
        ],
        f"单标签样本数: {data_stats['single_label_samples']}",
        f"双标签样本数: {data_stats['double_label_samples']}",
        f"三标签样本数: {data_stats['triple_label_samples']}",
        f"四标签样本数: {data_stats['quad_label_samples']}",
        f"平均每条样本标签数: {data_stats['avg_labels_per_sample']:.4f}",
        "",
        "评估指标:",
        f"micro_f1: {metrics['micro_f1']:.4f}",
        f"macro_f1: {metrics['macro_f1']:.4f}",
        f"weighted_f1: {metrics['weighted_f1']:.4f}",
        f"hamming_loss: {metrics['hamming_loss']:.4f}",
        f"subset_accuracy: {metrics['subset_accuracy']:.4f}",
        "",
        "classification_report:",
        cls_report,
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    stopwords_path = Path(args.stopwords)
    output_dir = Path(args.output_dir)

    report_path = output_dir / "multilabel_report.txt"
    predictions_path = output_dir / "multilabel_predictions.csv"
    model_path = output_dir / "multilabel_model.joblib"
    vectorizer_path = output_dir / "multilabel_tfidf_vectorizer.joblib"

    if not input_csv.exists():
        raise FileNotFoundError(f"未找到输入数据文件：{input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(input_csv)

    required_columns = ["text", *LABEL_COLUMNS, "labels"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要字段：{missing_columns}")

    stopwords = load_stopwords(stopwords_path)

    # 生成模型输入文本。这里先落到 DataFrame，方便后续保存与复查。
    df["text_processed"] = df["text"].apply(lambda value: preprocess_text(value, stopwords))

    y = df[LABEL_COLUMNS].astype(int)

    # 划分时直接保留原始 DataFrame，便于后续导出原文与真实标签。
    train_df, test_df, y_train, y_test = train_test_split(
        df,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # 只在训练集上拟合 TF-IDF，再用于测试集转换，避免数据泄漏。
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=1,
        max_df=0.95,
        token_pattern=r"(?u)\b\w+\b",
    )
    X_train = vectorizer.fit_transform(train_df["text_processed"])
    X_test = vectorizer.transform(test_df["text_processed"])

    classifier = OneVsRestClassifier(LogisticRegression(max_iter=2000, solver="liblinear"))
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=LABEL_COLUMNS, index=test_df.index)

    metrics = {
        "micro_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "hamming_loss": hamming_loss(y_test, y_pred),
        "subset_accuracy": accuracy_score(y_test, y_pred),
    }

    cls_report = classification_report(
        y_test,
        y_pred,
        target_names=LABEL_COLUMNS,
        digits=4,
        zero_division=0,
    )
    data_stats = build_data_stats(df, y, train_df, test_df)

    save_report(
        report_path=report_path,
        metrics=metrics,
        data_stats=data_stats,
        cls_report=cls_report,
        input_csv=input_csv,
        stopwords_path=stopwords_path,
    )

    predictions_df = pd.DataFrame(
        {
            "text": test_df["text"],
            "text_processed": test_df["text_processed"],
            "真实_labels": test_df["labels"],
            "预测_labels": y_pred_df.apply(labels_to_string, axis=1),
        },
        index=test_df.index,
    )

    for label in LABEL_COLUMNS:
        predictions_df[f"真实_{label}"] = test_df[label].astype(int)
    for label in LABEL_COLUMNS:
        predictions_df[f"预测_{label}"] = y_pred_df[label].astype(int)

    predictions_df = predictions_df.reset_index(drop=True)
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    joblib.dump(classifier, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("===== 多标签分类评估结果 =====")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("\n===== classification_report =====")
    print(cls_report)
    print(f"\n报告已保存到：{report_path}")
    print(f"预测结果已保存到：{predictions_path}")
    print(f"模型已保存到：{model_path}")
    print(f"向量器已保存到：{vectorizer_path}")


if __name__ == "__main__":
    main()
