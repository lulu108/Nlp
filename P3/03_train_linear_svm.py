#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
阶段3（重构版）：线性 SVM + dev 选参 + final test

功能：
1. 读取阶段2输出的 train/dev/test 特征与标签
2. 在 train 上训练，在 dev 上比较一组 C 候选
3. 选择 dev 上最佳参数
4. 用 train+dev 重训最佳模型
5. 最后只在 test 上评估一次
6. 保存模型、报告、预测结果与混淆矩阵图
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    MATPLOTLIB_IMPORT_ERROR = ""
except Exception as e:
    HAS_MATPLOTLIB = False
    MATPLOTLIB_IMPORT_ERROR = str(e)


def setup_matplotlib_for_zh() -> None:
    if not HAS_MATPLOTLIB:
        return

    # Use common Chinese fonts with fallback order to avoid tofu squares.
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

DEFAULT_X_TRAIN = DEFAULT_DATA_DIR / "X_train_tfidf.npz"
DEFAULT_X_DEV = DEFAULT_DATA_DIR / "X_dev_tfidf.npz"
DEFAULT_X_TEST = DEFAULT_DATA_DIR / "X_test_tfidf.npz"

DEFAULT_Y_TRAIN = DEFAULT_DATA_DIR / "y_train.npy"
DEFAULT_Y_DEV = DEFAULT_DATA_DIR / "y_dev.npy"
DEFAULT_Y_TEST = DEFAULT_DATA_DIR / "y_test.npy"

DEFAULT_TRAIN_CSV = DEFAULT_DATA_DIR / "train.csv"
DEFAULT_DEV_CSV = DEFAULT_DATA_DIR / "dev.csv"
DEFAULT_TEST_CSV = DEFAULT_DATA_DIR / "test.csv"

DEFAULT_MODEL_PATH = DEFAULT_DATA_DIR / "linear_svm_best_model.joblib"
DEFAULT_REPORT_PATH = DEFAULT_DATA_DIR / "linear_svm_3way_report.txt"
DEFAULT_PRED_CSV = DEFAULT_DATA_DIR / "linear_svm_test_predictions.csv"
DEFAULT_CM_PNG = DEFAULT_DATA_DIR / "linear_svm_test_confusion_matrix.png"

LABEL_NAMES = ["体育", "科技", "财经", "教育"]
LABEL_ID_TO_NAME = {0: "体育", 1: "科技", 2: "财经", 3: "教育"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="线性 SVM 三划分训练")
    parser.add_argument("--x-train", default=str(DEFAULT_X_TRAIN))
    parser.add_argument("--x-dev", default=str(DEFAULT_X_DEV))
    parser.add_argument("--x-test", default=str(DEFAULT_X_TEST))
    parser.add_argument("--y-train", default=str(DEFAULT_Y_TRAIN))
    parser.add_argument("--y-dev", default=str(DEFAULT_Y_DEV))
    parser.add_argument("--y-test", default=str(DEFAULT_Y_TEST))
    parser.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--dev-csv", default=str(DEFAULT_DEV_CSV))
    parser.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--pred-out", default=str(DEFAULT_PRED_CSV))
    parser.add_argument("--cm-out", default=str(DEFAULT_CM_PNG))
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0], help="待比较的C参数列表")
    parser.add_argument("--max-iter", type=int, default=8000)
    parser.add_argument(
        "--auto-dev-size",
        type=float,
        default=0.2,
        help="当 dev 特征文件不存在时，从 train 自动切分 dev 的比例",
    )
    parser.add_argument(
        "--auto-dev-random-state",
        type=int,
        default=42,
        help="自动切分 dev 时的随机种子",
    )
    parser.add_argument(
        "--allow-auto-dev-split",
        action="store_true",
        help="允许在缺少 dev 文件时，从 train 自动切分 dev。默认关闭（推荐严格使用阶段2产物）。",
    )
    return parser.parse_args()


def save_confusion_matrix_figure(cm: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    if not HAS_MATPLOTLIB:
        print(
            "[提示] matplotlib 不可用，跳过混淆矩阵图片保存。"
            f"错误: {MATPLOTLIB_IMPORT_ERROR}"
        )
        return

    setup_matplotlib_for_zh()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "acc": acc,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "weighted_p": weighted_p,
        "weighted_r": weighted_r,
        "weighted_f1": weighted_f1,
        "report": report,
        "cm": cm,
    }


def main() -> None:
    args = parse_args()

    x_train_path = Path(args.x_train)
    x_dev_path = Path(args.x_dev)
    y_train_path = Path(args.y_train)
    y_dev_path = Path(args.y_dev)

    X_train = load_npz(x_train_path)
    X_test = load_npz(Path(args.x_test))

    y_train = np.load(y_train_path)
    y_test = np.load(Path(args.y_test))

    auto_dev_split_used = False
    if x_dev_path.exists() and y_dev_path.exists():
        X_dev = load_npz(x_dev_path)
        y_dev = np.load(y_dev_path)
    else:
        if not args.allow_auto_dev_split:
            raise FileNotFoundError(
                "未找到 dev 特征文件，请先运行阶段2三划分脚本生成：\n"
                f"- {x_dev_path}\n"
                f"- {y_dev_path}\n"
                "若你确实需要临时从 train 自动切分 dev，请增加参数 --allow-auto-dev-split"
            )

        auto_dev_split_used = True
        idx_all = np.arange(y_train.shape[0])
        idx_train_new, idx_dev_new = train_test_split(
            idx_all,
            test_size=args.auto_dev_size,
            random_state=args.auto_dev_random_state,
            stratify=y_train,
            shuffle=True,
        )

        X_dev = X_train[idx_dev_new]
        y_dev = y_train[idx_dev_new]
        X_train = X_train[idx_train_new]
        y_train = y_train[idx_train_new]

        print(
            "[提示] 未找到 dev 特征文件，已从训练集自动切分 dev。"
            f"dev_size={args.auto_dev_size}, random_state={args.auto_dev_random_state}"
        )

    train_csv_path = Path(args.train_csv)
    dev_csv_path = Path(args.dev_csv)
    test_csv_path = Path(args.test_csv)

    # train/dev csv 仅用于必要时排查，当前流程只依赖 test_df 生成预测对照表。
    train_df = pd.read_csv(train_csv_path, encoding="utf-8-sig") if train_csv_path.exists() else pd.DataFrame()
    dev_df = pd.read_csv(dev_csv_path, encoding="utf-8-sig") if dev_csv_path.exists() else pd.DataFrame()
    test_df = pd.read_csv(test_csv_path, encoding="utf-8-sig")

    print("===== 阶段3：LinearSVC 三划分训练开始 =====")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"X_test shape: {X_test.shape}")

    # 1. 用 dev 选参数
    dev_results = []
    best_c = None
    best_score = -1.0
    best_dev_model = None

    print("\n[1] 在 dev 上比较参数...")
    for c in args.c_grid:
        clf = LinearSVC(C=c, max_iter=args.max_iter)
        clf.fit(X_train, y_train)
        y_dev_pred = clf.predict(X_dev)
        metrics = evaluate(y_dev, y_dev_pred)

        dev_results.append({
            "C": c,
            "dev_acc": metrics["acc"],
            "dev_macro_f1": metrics["macro_f1"],
            "dev_macro_p": metrics["macro_p"],
            "dev_macro_r": metrics["macro_r"],
        })

        print(f"C={c:.4f} | dev_acc={metrics['acc']:.4f} | dev_macro_f1={metrics['macro_f1']:.4f}")

        # 以 dev_macro_f1 作为主要选模指标
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_c = c
            best_dev_model = clf

    print(f"\n最优参数: C={best_c}，最佳 dev_macro_f1={best_score:.4f}")

    # 2. 用 train+dev 重训最佳模型
    print("\n[2] 使用 train+dev 重训最佳模型...")
    X_train_dev = vstack([X_train, X_dev])
    y_train_dev = np.concatenate([y_train, y_dev], axis=0)

    final_clf = LinearSVC(C=best_c, max_iter=args.max_iter)
    final_clf.fit(X_train_dev, y_train_dev)

    # 3. 只在 test 上评估一次
    print("[3] 在 test 上做最终评估...")
    y_test_pred = final_clf.predict(X_test)
    final_metrics = evaluate(y_test, y_test_pred)

    # 4. 保存模型
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, model_out)

    # 5. 保存 test 预测结果
    pred_out = Path(args.pred_out)
    pred_df = test_df.copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_test_pred
    pred_df["true_category"] = pred_df["y_true"].map(LABEL_ID_TO_NAME)
    pred_df["pred_category"] = pred_df["y_pred"].map(LABEL_ID_TO_NAME)
    pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")

    # 6. 保存混淆矩阵图
    cm_out = Path(args.cm_out)
    if HAS_MATPLOTLIB:
        save_confusion_matrix_figure(
            final_metrics["cm"],
            LABEL_NAMES,
            cm_out,
            title="Linear SVM Test Confusion Matrix",
        )
        cm_output_note = f"混淆矩阵图: {cm_out}"
    else:
        cm_output_note = (
            "混淆矩阵图: 已跳过（matplotlib 不可用，"
            f"错误={MATPLOTLIB_IMPORT_ERROR}）"
        )

    # 7. 保存报告
    report_out = Path(args.report_out)
    report_lines = []
    report_lines.append("===== LinearSVC 三划分实验报告 =====")
    report_lines.append("")
    report_lines.append("[1] 数据规模")
    report_lines.append(f"X_train shape = {X_train.shape}")
    report_lines.append(f"X_dev shape = {X_dev.shape}")
    report_lines.append(f"X_test shape = {X_test.shape}")
    report_lines.append(f"auto_dev_split_used = {auto_dev_split_used}")
    report_lines.append("")
    report_lines.append("[2] Dev 参数选择结果")
    for item in dev_results:
        report_lines.append(
            f"C={item['C']}, dev_acc={item['dev_acc']:.4f}, "
            f"dev_macro_f1={item['dev_macro_f1']:.4f}, "
            f"dev_macro_p={item['dev_macro_p']:.4f}, "
            f"dev_macro_r={item['dev_macro_r']:.4f}"
        )
    report_lines.append("")
    report_lines.append("[3] 最优参数")
    report_lines.append(f"best_C = {best_c}")
    report_lines.append(f"best_dev_macro_f1 = {best_score:.4f}")
    report_lines.append("")
    report_lines.append("[4] Test 最终结果")
    report_lines.append(f"Accuracy = {final_metrics['acc']:.4f}")
    report_lines.append(f"Macro Precision = {final_metrics['macro_p']:.4f}")
    report_lines.append(f"Macro Recall = {final_metrics['macro_r']:.4f}")
    report_lines.append(f"Macro F1 = {final_metrics['macro_f1']:.4f}")
    report_lines.append(f"Weighted Precision = {final_metrics['weighted_p']:.4f}")
    report_lines.append(f"Weighted Recall = {final_metrics['weighted_r']:.4f}")
    report_lines.append(f"Weighted F1 = {final_metrics['weighted_f1']:.4f}")
    report_lines.append("")
    report_lines.append("[5] 分类报告")
    report_lines.append(final_metrics["report"])
    report_lines.append("")
    report_lines.append("[6] 混淆矩阵")
    report_lines.append(str(final_metrics["cm"]))
    report_lines.append("")
    report_lines.append("[7] 输出文件")
    report_lines.append(f"模型文件: {model_out}")
    report_lines.append(f"预测结果: {pred_out}")
    report_lines.append(cm_output_note)

    report_out.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n===== Test 最终结果 =====")
    print(f"Accuracy = {final_metrics['acc']:.4f}")
    print(f"Macro Precision = {final_metrics['macro_p']:.4f}")
    print(f"Macro Recall = {final_metrics['macro_r']:.4f}")
    print(f"Macro F1 = {final_metrics['macro_f1']:.4f}")
    print("\n分类报告：")
    print(final_metrics["report"])
    print("混淆矩阵：")
    print(final_metrics["cm"])

    print("\n===== 输出文件 =====")
    print(f"模型文件: {model_out}")
    print(f"报告文件: {report_out}")
    print(f"预测结果: {pred_out}")
    print(cm_output_note)


if __name__ == "__main__":
    main()