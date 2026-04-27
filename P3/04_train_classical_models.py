"""
阶段4：传统机器学习多模型对比实验
支持模型：
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM

流程：
- 读取 train/dev/test 特征
- 在 dev 上调参
- 选最佳参数
- 用 train+dev 重训
- 在 test 上最终评估
- 保存模型、预测结果、混淆矩阵、汇总表与总报告
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    MATPLOTLIB_IMPORT_ERROR = ""
except Exception as e:
    HAS_MATPLOTLIB = False
    MATPLOTLIB_IMPORT_ERROR = str(e)


LABEL_NAMES = ["体育", "科技", "财经", "教育"]
LABEL_ID_TO_NAME = {0: "体育", 1: "科技", 2: "财经", 3: "教育"}


def setup_matplotlib_for_zh() -> None:
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = data_dir / "classical_model_outputs"

    parser = argparse.ArgumentParser(description="传统机器学习多模型对比实验")
    parser.add_argument("--x-train", default=str(data_dir / "X_train_tfidf.npz"))
    parser.add_argument("--x-dev", default=str(data_dir / "X_dev_tfidf.npz"))
    parser.add_argument("--x-test", default=str(data_dir / "X_test_tfidf.npz"))
    parser.add_argument("--y-train", default=str(data_dir / "y_train.npy"))
    parser.add_argument("--y-dev", default=str(data_dir / "y_dev.npy"))
    parser.add_argument("--y-test", default=str(data_dir / "y_test.npy"))

    parser.add_argument("--train-csv", default=str(data_dir / "train.csv"))
    parser.add_argument("--dev-csv", default=str(data_dir / "dev.csv"))
    parser.add_argument("--test-csv", default=str(data_dir / "test.csv"))

    parser.add_argument("--output-dir", default=str(output_dir))
    parser.add_argument("--report-out", default=str(output_dir / "classical_models_report.txt"))
    parser.add_argument("--summary-out", default=str(output_dir / "classical_models_summary.csv"))

    parser.add_argument("--nb-alpha-grid", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--lr-c-grid", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--svm-c-grid", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0])

    parser.add_argument("--lr-max-iter", type=int, default=5000)
    parser.add_argument("--svm-max-iter", type=int, default=8000)

    parser.add_argument("--auto-dev-size", type=float, default=0.2)
    parser.add_argument("--auto-dev-random-state", type=int, default=42)

    return parser.parse_args()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
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


def save_confusion_matrix_figure(cm: np.ndarray, labels: List[str], out_path: Path, title: str) -> None:
    if not HAS_MATPLOTLIB:
        print(f"[提示] matplotlib 不可用，跳过混淆矩阵图保存：{MATPLOTLIB_IMPORT_ERROR}")
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


def load_data(args: argparse.Namespace):
    x_train_path = Path(args.x_train)
    x_dev_path = Path(args.x_dev)
    x_test_path = Path(args.x_test)
    y_train_path = Path(args.y_train)
    y_dev_path = Path(args.y_dev)
    y_test_path = Path(args.y_test)

    X_train = load_npz(x_train_path)
    X_test = load_npz(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    auto_dev_split_used = False

    if x_dev_path.exists() and y_dev_path.exists():
        X_dev = load_npz(x_dev_path)
        y_dev = np.load(y_dev_path)
    else:
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

    return X_train, y_train, X_dev, y_dev, X_test, y_test, auto_dev_split_used


def load_csv_safe(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()


def build_model_candidates(args: argparse.Namespace) -> Dict[str, List[Tuple[str, object]]]:
    candidates = {
        "NB": [],
        "LR": [],
        "SVM": [],
    }

    for alpha in args.nb_alpha_grid:
        model = MultinomialNB(alpha=alpha)
        candidates["NB"].append((f"alpha={alpha}", model))

    for c in args.lr_c_grid:
        base_lr = LogisticRegression(
            C=c,
            max_iter=args.lr_max_iter,
            solver="liblinear",
        )
        model = OneVsRestClassifier(base_lr)
        candidates["LR"].append((f"C={c}", model))

    for c in args.svm_c_grid:
        model = LinearSVC(
            C=c,
            max_iter=args.svm_max_iter,
        )
        candidates["SVM"].append((f"C={c}", model))

    return candidates


def train_select_and_test(
    model_name: str,
    model_candidates: List[Tuple[str, object]],
    X_train,
    y_train,
    X_dev,
    y_dev,
    X_test,
    y_test,
    test_df: pd.DataFrame,
    output_dir: Path,
    lr_max_iter: int,
    svm_max_iter: int,
) -> Dict:
    print(f"\n===== 开始模型：{model_name} =====")

    best_score = -1.0
    best_param_desc = None
    best_model = None
    dev_rows = []

    # 1. dev 选参
    for param_desc, model in model_candidates:
        model.fit(X_train, y_train)
        y_dev_pred = model.predict(X_dev)
        metrics = evaluate(y_dev, y_dev_pred)

        row = {
            "model": model_name,
            "param": param_desc,
            "dev_acc": metrics["acc"],
            "dev_macro_p": metrics["macro_p"],
            "dev_macro_r": metrics["macro_r"],
            "dev_macro_f1": metrics["macro_f1"],
        }
        dev_rows.append(row)

        print(
            f"{model_name} | {param_desc} | "
            f"dev_acc={metrics['acc']:.4f} | dev_macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_param_desc = param_desc
            best_model = model

    print(f"[{model_name}] 最优参数: {best_param_desc}, best_dev_macro_f1={best_score:.4f}")

    # 2. 用 train+dev 重训最佳模型
    X_train_dev = vstack([X_train, X_dev])
    y_train_dev = np.concatenate([y_train, y_dev], axis=0)

    # 重新根据 best_param_desc 构造模型，避免复用已训练状态对象
    if model_name == "NB":
        alpha = float(best_param_desc.split("=")[1])
        final_model = MultinomialNB(alpha=alpha)
    elif model_name == "LR":
        c = float(best_param_desc.split("=")[1])
        final_model = OneVsRestClassifier(
            LogisticRegression(
                C=c,
                max_iter=lr_max_iter,
                solver="liblinear",
            )
        )
    elif model_name == "SVM":
        c = float(best_param_desc.split("=")[1])
        final_model = LinearSVC(
            C=c,
            max_iter=svm_max_iter,
        )
    else:
        raise ValueError(f"未知模型名: {model_name}")

    final_model.fit(X_train_dev, y_train_dev)

    # 3. test 最终评估
    y_test_pred = final_model.predict(X_test)
    test_metrics = evaluate(y_test, y_test_pred)

    # 4. 保存模型
    model_path = output_dir / f"{model_name.lower()}_best_model.joblib"
    joblib.dump(final_model, model_path)

    # 5. 保存预测结果
    pred_df = test_df.copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_test_pred
    pred_df["true_category"] = pd.Series(y_test).map(LABEL_ID_TO_NAME)
    pred_df["pred_category"] = pd.Series(y_test_pred).map(LABEL_ID_TO_NAME)

    pred_path = output_dir / f"{model_name.lower()}_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 6. 保存混淆矩阵图
    cm_path = output_dir / f"{model_name.lower()}_test_confusion_matrix.png"
    save_confusion_matrix_figure(
        test_metrics["cm"],
        LABEL_NAMES,
        cm_path,
        title=f"{model_name} Test Confusion Matrix",
    )

    # 7. 保存分类报告
    cls_report_path = output_dir / f"{model_name.lower()}_classification_report.txt"
    cls_report_path.write_text(test_metrics["report"], encoding="utf-8")

    return {
        "model": model_name,
        "best_param": best_param_desc,
        "best_dev_macro_f1": best_score,
        "test_acc": test_metrics["acc"],
        "test_macro_p": test_metrics["macro_p"],
        "test_macro_r": test_metrics["macro_r"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_report": test_metrics["report"],
        "test_cm": test_metrics["cm"],
        "model_path": str(model_path),
        "pred_path": str(pred_path),
        "cm_path": str(cm_path),
        "cls_report_path": str(cls_report_path),
        "dev_rows": dev_rows,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_dev, y_dev, X_test, y_test, auto_dev_split_used = load_data(args)

    train_df = load_csv_safe(args.train_csv)
    dev_df = load_csv_safe(args.dev_csv)
    test_df = load_csv_safe(args.test_csv)

    print("===== 多模型对比实验开始 =====")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"auto_dev_split_used: {auto_dev_split_used}")

    candidates = build_model_candidates(args)

    all_results = []
    all_dev_rows = []

    for model_name in ["NB", "LR", "SVM"]:
        result = train_select_and_test(
            model_name=model_name,
            model_candidates=candidates[model_name],
            X_train=X_train,
            y_train=y_train,
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            y_test=y_test,
            test_df=test_df,
            output_dir=output_dir,
            lr_max_iter=args.lr_max_iter,
            svm_max_iter=args.svm_max_iter,
        )
        all_results.append(result)
        all_dev_rows.extend(result["dev_rows"])

    # 汇总表
    summary_df = pd.DataFrame([
        {
            "model": r["model"],
            "best_param": r["best_param"],
            "best_dev_macro_f1": r["best_dev_macro_f1"],
            "test_acc": r["test_acc"],
            "test_macro_p": r["test_macro_p"],
            "test_macro_r": r["test_macro_r"],
            "test_macro_f1": r["test_macro_f1"],
            "test_weighted_f1": r["test_weighted_f1"],
        }
        for r in all_results
    ]).sort_values(by="test_macro_f1", ascending=False)

    summary_path = Path(args.summary_out)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # dev参数搜索表
    dev_detail_df = pd.DataFrame(all_dev_rows)
    dev_detail_path = output_dir / "dev_search_details.csv"
    dev_detail_df.to_csv(dev_detail_path, index=False, encoding="utf-8-sig")

    # 总报告
    report_lines = []
    report_lines.append("===== 传统机器学习多模型对比实验报告 =====")
    report_lines.append("")
    report_lines.append("[1] 数据规模")
    report_lines.append(f"X_train shape = {X_train.shape}")
    report_lines.append(f"X_dev shape = {X_dev.shape}")
    report_lines.append(f"X_test shape = {X_test.shape}")
    report_lines.append(f"auto_dev_split_used = {auto_dev_split_used}")
    report_lines.append("")

    report_lines.append("[2] Dev参数搜索结果")
    for row in all_dev_rows:
        report_lines.append(
            f"{row['model']} | {row['param']} | "
            f"dev_acc={row['dev_acc']:.4f} | "
            f"dev_macro_p={row['dev_macro_p']:.4f} | "
            f"dev_macro_r={row['dev_macro_r']:.4f} | "
            f"dev_macro_f1={row['dev_macro_f1']:.4f}"
        )
    report_lines.append("")

    report_lines.append("[3] Test最终结果汇总")
    report_lines.append(summary_df.to_string(index=False))
    report_lines.append("")

    for r in all_results:
        report_lines.append(f"[4] 模型：{r['model']}")
        report_lines.append(f"best_param = {r['best_param']}")
        report_lines.append(f"best_dev_macro_f1 = {r['best_dev_macro_f1']:.4f}")
        report_lines.append(f"test_acc = {r['test_acc']:.4f}")
        report_lines.append(f"test_macro_p = {r['test_macro_p']:.4f}")
        report_lines.append(f"test_macro_r = {r['test_macro_r']:.4f}")
        report_lines.append(f"test_macro_f1 = {r['test_macro_f1']:.4f}")
        report_lines.append(f"test_weighted_f1 = {r['test_weighted_f1']:.4f}")
        report_lines.append("分类报告：")
        report_lines.append(r["test_report"])
        report_lines.append("混淆矩阵：")
        report_lines.append(str(r["test_cm"]))
        report_lines.append("")
        report_lines.append(f"模型文件: {r['model_path']}")
        report_lines.append(f"预测结果: {r['pred_path']}")
        report_lines.append(f"混淆矩阵图: {r['cm_path']}")
        report_lines.append(f"分类报告文件: {r['cls_report_path']}")
        report_lines.append("")

    report_lines.append("[5] 汇总文件")
    report_lines.append(f"summary_csv = {summary_path}")
    report_lines.append(f"dev_search_details_csv = {dev_detail_path}")

    report_out = Path(args.report_out)
    report_out.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n===== Test最终结果汇总 =====")
    print(summary_df.to_string(index=False))

    print("\n===== 输出文件 =====")
    print(f"总报告: {report_out}")
    print(f"结果汇总表: {summary_path}")
    print(f"dev搜索明细: {dev_detail_path}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()