# -*- coding: utf-8 -*-
"""
统一汇总 P1 实验链路指标，输出：
- P1/reports/experiment_summary.md
- P1/reports/metrics_summary.csv

说明：
1. 仅解析现有日志/结果文件，不触发训练。
2. 对缺失或无法解析的指标统一标记为"待补充"。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
P1_DIR = ROOT_DIR / "P1"
REPORT_DIR = P1_DIR / "reports"

SUMMARY_MD = REPORT_DIR / "experiment_summary.md"
SUMMARY_CSV = REPORT_DIR / "metrics_summary.csv"


def read_text_auto(path: Path) -> str:
    if not path.exists():
        return ""
    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "gbk"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def parse_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def fmt_metric(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "待补充"
    return f"{value:.{precision}f}"


def count_non_empty_lines(path: Path) -> int:
    txt = read_text_auto(path)
    if not txt:
        return 0
    return len([line for line in txt.splitlines() if line.strip()])


def parse_tag_table_macro_f1(log_text: str, section_title: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pattern = re.compile(
        rf"\[{re.escape(section_title)}\]\s*各标签 P/R/F1\s*\n"
        r"Tag\s+Precision\s+Recall\s+F1\s*\n"
        r"(.*?)\n\n",
        flags=re.DOTALL,
    )
    m = pattern.search(log_text)
    if not m:
        return None, None, None

    body = m.group(1)
    ps: List[float] = []
    rs: List[float] = []
    f1s: List[float] = []

    for line in body.splitlines():
        cols = line.split()
        if len(cols) < 4:
            continue
        try:
            ps.append(float(cols[-3]))
            rs.append(float(cols[-2]))
            f1s.append(float(cols[-1]))
        except Exception:
            continue

    if not ps:
        return None, None, None

    return sum(ps) / len(ps), sum(rs) / len(rs), sum(f1s) / len(f1s)


def parse_per_tag_metrics(log_text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    rows = re.findall(r"^[BMES]:\s*P=([0-9.]+)\s*R=([0-9.]+)\s*F1=([0-9.]+)", log_text, flags=re.MULTILINE)
    if not rows:
        return None, None, None

    ps = [float(x[0]) for x in rows]
    rs = [float(x[1]) for x in rows]
    f1s = [float(x[2]) for x in rows]
    return sum(ps) / len(ps), sum(rs) / len(rs), sum(f1s) / len(f1s)


def parse_confusion_tsv(path: Path) -> Optional[List[List[int]]]:
    txt = read_text_auto(path)
    if not txt:
        return None

    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    matrix: List[List[int]] = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        try:
            nums = [int(x) for x in cols[1:5]]
        except Exception:
            return None
        matrix.append(nums)

    if len(matrix) != 4:
        return None
    return matrix


def metrics_from_confusion(confusion: List[List[int]]) -> Tuple[float, float, float, float]:
    total = sum(sum(r) for r in confusion)
    correct = sum(confusion[i][i] for i in range(4))
    acc = (correct / total) if total else 0.0

    p_list: List[float] = []
    r_list: List[float] = []
    f1_list: List[float] = []

    for i in range(4):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(4)) - tp
        fn = sum(confusion[i][c] for c in range(4)) - tp

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)

    return acc, sum(p_list) / 4.0, sum(r_list) / 4.0, sum(f1_list) / 4.0


def parse_hmm_main() -> Dict[str, str]:
    latest_main_log = P1_DIR / "logs" / "hmm_main_latest.txt"
    latest_main_text = read_text_auto(latest_main_log)

    if latest_main_text:
        train_samples = parse_int(r"训练集样本数[:：]\s*(\d+)", latest_main_text)
        test_samples = parse_int(r"测试集样本数[:：]\s*(\d+)", latest_main_text)
        tag_acc = parse_float(r"测试集标签级准确率[:：]\s*([0-9.]+)", latest_main_text)
        macro_p, macro_r, macro_f1 = parse_per_tag_metrics(latest_main_text)

        return {
            "实验链路": "P1 主线 HMM",
            "标签准确率": fmt_metric(tag_acc),
            "标签级Macro Precision": fmt_metric(macro_p),
            "标签级Macro Recall": fmt_metric(macro_r),
            "标签级Macro F1": fmt_metric(macro_f1),
            "token数": "待补充",
            "PER实体数": "待补充",
            "LOC实体数": "待补充",
            "ORG实体数": "待补充",
            "交叉验证Mean": "待补充",
            "交叉验证Std": "待补充",
            "训练样本数": str(train_samples) if train_samples is not None else "待补充",
            "测试样本数": str(test_samples) if test_samples is not None else "待补充",
            "指标来源": "P1/logs/hmm_main_latest.txt",
            "备注": "优先解析最新主线 HMM 控制台日志",
        }

    split_log = read_text_auto(P1_DIR / "logs" / "three_split_after_reorg.txt")
    eval_log = read_text_auto(P1_DIR / "logs" / "eval_plot_base_env.txt")

    train_samples = parse_int(r"训练集样本数[:：]\s*(\d+)", split_log)
    test_samples = parse_int(r"最终测试集样本数[:：]\s*(\d+)", split_log)

    tag_acc = parse_float(r"\[Test Final\]\s*标签级准确率[:：]\s*([0-9.]+)", split_log)
    if tag_acc is None:
        tag_acc = parse_float(r"\[Auto Test\]\s*标签级准确率[:：]\s*([0-9.]+)", eval_log)

    macro_p, macro_r, macro_f1 = parse_tag_table_macro_f1(split_log, "Test Final")

    cv_mean = parse_float(r"Mean[:：]\s*([0-9.]+)", split_log)
    cv_std = parse_float(r"Std[:：]\s*([0-9.]+)", split_log)

    notes = []
    if split_log:
        notes.append("优先解析 logs/three_split_after_reorg.txt")
    elif eval_log:
        notes.append("回退解析 logs/eval_plot_base_env.txt")
    else:
        notes.append("缺少可解析日志")

    return {
        "实验链路": "P1 主线 HMM",
        "标签准确率": fmt_metric(tag_acc),
        "标签级Macro Precision": fmt_metric(macro_p),
        "标签级Macro Recall": fmt_metric(macro_r),
        "标签级Macro F1": fmt_metric(macro_f1),
        "token数": "待补充",
        "PER实体数": "待补充",
        "LOC实体数": "待补充",
        "ORG实体数": "待补充",
        "交叉验证Mean": fmt_metric(cv_mean),
        "交叉验证Std": fmt_metric(cv_std),
        "训练样本数": str(train_samples) if train_samples is not None else "待补充",
        "测试样本数": str(test_samples) if test_samples is not None else "待补充",
        "指标来源": "P1/logs/three_split_after_reorg.txt; P1/logs/eval_plot_base_env.txt",
        "备注": "；".join(notes),
    }


def parse_hmm_test2() -> Dict[str, str]:
    metrics_path = P1_DIR / "test2" / "output" / "metrics.json"
    if metrics_path.exists():
        raw = read_text_auto(metrics_path)
        try:
            metrics = json.loads(raw)
        except Exception:
            metrics = {}

        tag_acc = metrics.get("tag_accuracy")
        macro_p = metrics.get("macro_precision")
        macro_r = metrics.get("macro_recall")
        macro_f1 = metrics.get("macro_f1")
        train_count = metrics.get("train_size")
        test_count = metrics.get("test_size")
        confusion_path = metrics.get("confusion_matrix_path", "P1/test2/output/confusion_matrix.tsv")
        label_report_path = metrics.get("label_report_path", "P1/test2/output/label_report.tsv")

        return {
            "实验链路": "P1/test2 迭代版 HMM",
            "标签准确率": fmt_metric(tag_acc),
            "标签级Macro Precision": fmt_metric(macro_p),
            "标签级Macro Recall": fmt_metric(macro_r),
            "标签级Macro F1": fmt_metric(macro_f1),
            "token数": "待补充",
            "PER实体数": "待补充",
            "LOC实体数": "待补充",
            "ORG实体数": "待补充",
            "交叉验证Mean": "待补充",
            "交叉验证Std": "待补充",
            "训练样本数": str(train_count) if train_count is not None else "待补充",
            "测试样本数": str(test_count) if test_count is not None else "待补充",
            "指标来源": f"{metrics_path}; {confusion_path}; {label_report_path}",
            "备注": "已生成 test2 结构化评估输出 (metrics.json / confusion_matrix / label_report)",
            "混淆矩阵路径": confusion_path,
            "标签报告路径": label_report_path,
            "has_metrics": True,
        }

    train_count = count_non_empty_lines(P1_DIR / "test2" / "datasets" / "auto" / "train.txt")
    test_count = count_non_empty_lines(P1_DIR / "test2" / "datasets" / "auto" / "test.txt")

    return {
        "实验链路": "P1/test2 迭代版 HMM",
        "标签准确率": "待补充",
        "标签级Macro Precision": "待补充",
        "标签级Macro Recall": "待补充",
        "标签级Macro F1": "待补充",
        "token数": "待补充",
        "PER实体数": "待补充",
        "LOC实体数": "待补充",
        "ORG实体数": "待补充",
        "交叉验证Mean": "待补充",
        "交叉验证Std": "待补充",
        "训练样本数": str(train_count) if train_count else "待补充",
        "测试样本数": str(test_count) if test_count else "待补充",
        "指标来源": "P1/test2/output/top_error_words.txt; P1/test2/output/train_top_errors.txt",
        "备注": "当前输出以错误分析为主，缺少可直接解析的准确率/F1 数值日志",
        "混淆矩阵路径": "待补充",
        "标签报告路径": "待补充",
        "has_metrics": False,
    }


def parse_label_report_m_metrics(path_text: str) -> Dict[str, str]:
    path = Path(path_text)
    txt = read_text_auto(path)
    if not txt:
        return {"m_recall": "待补充", "m_f1": "待补充"}

    for line in txt.splitlines():
        cols = line.strip().split("\t")
        if len(cols) < 5:
            continue
        if cols[0] != "M":
            continue
        try:
            recall = float(cols[2])
            f1 = float(cols[3])
        except Exception:
            return {"m_recall": "待补充", "m_f1": "待补充"}
        return {"m_recall": fmt_metric(recall), "m_f1": fmt_metric(f1)}

    return {"m_recall": "待补充", "m_f1": "待补充"}


def parse_nlp4j_baseline() -> Dict[str, str]:
    result_path = P1_DIR / "nlp4j_baseline" / "output" / "nlp4j_result.tsv"
    txt = read_text_auto(result_path)

    token_count = None
    per_count = 0
    loc_count = 0
    org_count = 0

    if txt:
        data_lines = []
        for raw_line in txt.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("sentence_id\t"):
                continue
            data_lines.append(line)

        valid_rows = 0
        for line in data_lines:
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            _, _token, _pos, entity = parts
            valid_rows += 1
            if entity == "PER":
                per_count += 1
            elif entity == "LOC":
                loc_count += 1
            elif entity == "ORG":
                org_count += 1

        token_count = valid_rows if valid_rows > 0 else None

    has_result = token_count is not None
    if has_result:
        note = "词典规则型 NLP4J baseline，非训练式模型指标；Accuracy/F1 仍待补充"
    else:
        note = "词典规则型 NLP4J baseline 尚未生成结果；Accuracy/F1 待补充"

    return {
        "实验链路": "P1/nlp4j_baseline NLP4J 对照实验",
        "标签准确率": "待补充",
        "标签级Macro Precision": "待补充",
        "标签级Macro Recall": "待补充",
        "标签级Macro F1": "待补充",
        "token数": str(token_count) if token_count is not None else "待补充",
        "PER实体数": str(per_count) if has_result else "待补充",
        "LOC实体数": str(loc_count) if has_result else "待补充",
        "ORG实体数": str(org_count) if has_result else "待补充",
        "交叉验证Mean": "待补充",
        "交叉验证Std": "待补充",
        "训练样本数": "待补充",
        "测试样本数": "待补充",
        "指标来源": "P1/nlp4j_baseline/output/nlp4j_result.tsv",
        "备注": note,
    }


def parse_nlp4j_hmm() -> Dict[str, str]:
    metrics_path = P1_DIR / "nlp4j_baseline" / "output" / "nlp4j_hmm_metrics.json"
    if not metrics_path.exists():
        return {
            "实验链路": "P1/nlp4j_baseline Java HMM 训练式基线",
            "标签准确率": "待补充",
            "标签级Macro Precision": "待补充",
            "标签级Macro Recall": "待补充",
            "标签级Macro F1": "待补充",
            "token数": "待补充",
            "PER实体数": "待补充",
            "LOC实体数": "待补充",
            "ORG实体数": "待补充",
            "交叉验证Mean": "待补充",
            "交叉验证Std": "待补充",
            "训练样本数": "待补充",
            "测试样本数": "待补充",
            "指标来源": "P1/nlp4j_baseline/output/nlp4j_hmm_metrics.json",
            "备注": "尚未生成训练式 Java HMM 指标",
        }

    raw = read_text_auto(metrics_path)
    if not raw:
        return {
            "实验链路": "P1/nlp4j_baseline Java HMM 训练式基线",
            "标签准确率": "待补充",
            "标签级Macro Precision": "待补充",
            "标签级Macro Recall": "待补充",
            "标签级Macro F1": "待补充",
            "token数": "待补充",
            "PER实体数": "待补充",
            "LOC实体数": "待补充",
            "ORG实体数": "待补充",
            "交叉验证Mean": "待补充",
            "交叉验证Std": "待补充",
            "训练样本数": "待补充",
            "测试样本数": "待补充",
            "指标来源": "P1/nlp4j_baseline/output/nlp4j_hmm_metrics.json",
            "备注": "训练式 Java HMM metrics.json 为空或无法解析",
        }

    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    train_size = data.get("train_size")
    test_size = data.get("test_size")
    return {
        "实验链路": "P1/nlp4j_baseline Java HMM 训练式基线",
        "标签准确率": fmt_metric(data.get("tag_accuracy")),
        "标签级Macro Precision": fmt_metric(data.get("macro_precision")),
        "标签级Macro Recall": fmt_metric(data.get("macro_recall")),
        "标签级Macro F1": fmt_metric(data.get("macro_f1")),
        "token数": "待补充",
        "PER实体数": "待补充",
        "LOC实体数": "待补充",
        "ORG实体数": "待补充",
        "交叉验证Mean": "待补充",
        "交叉验证Std": "待补充",
        "训练样本数": str(train_size) if train_size is not None else "待补充",
        "测试样本数": str(test_size) if test_size is not None else "待补充",
        "指标来源": "P1/nlp4j_baseline/output/nlp4j_hmm_metrics.json",
        "备注": "训练式 Java HMM BMES baseline，非官方 NLP4J 训练模型",
    }


def parse_bilstm_crf() -> Dict[str, str]:
    confusion_path = P1_DIR / "BiLSTMCRF" / "output" / "confusion_matrix.tsv"
    confusion = parse_confusion_tsv(confusion_path)

    train_count = count_non_empty_lines(P1_DIR / "BiLSTMCRF" / "data" / "train.txt")
    test_count = count_non_empty_lines(P1_DIR / "BiLSTMCRF" / "data" / "test.txt")

    if confusion is None:
        return {
            "实验链路": "P1/BiLSTMCRF 深度学习版本",
            "标签准确率": "待补充",
            "标签级Macro Precision": "待补充",
            "标签级Macro Recall": "待补充",
            "标签级Macro F1": "待补充",
            "token数": "待补充",
            "PER实体数": "待补充",
            "LOC实体数": "待补充",
            "ORG实体数": "待补充",
            "交叉验证Mean": "待补充",
            "交叉验证Std": "待补充",
            "训练样本数": str(train_count) if train_count else "待补充",
            "测试样本数": str(test_count) if test_count else "待补充",
            "指标来源": "P1/BiLSTMCRF/output/confusion_matrix.tsv",
            "备注": "未解析到有效混淆矩阵",
        }

    acc, p, r, f1 = metrics_from_confusion(confusion)
    return {
        "实验链路": "P1/BiLSTMCRF 深度学习版本",
        "标签准确率": fmt_metric(acc),
        "标签级Macro Precision": fmt_metric(p),
        "标签级Macro Recall": fmt_metric(r),
        "标签级Macro F1": fmt_metric(f1),
        "token数": "待补充",
        "PER实体数": "待补充",
        "LOC实体数": "待补充",
        "ORG实体数": "待补充",
        "交叉验证Mean": "待补充",
        "交叉验证Std": "待补充",
        "训练样本数": str(train_count) if train_count else "待补充",
        "测试样本数": str(test_count) if test_count else "待补充",
        "指标来源": "P1/BiLSTMCRF/output/confusion_matrix.tsv",
        "备注": "由混淆矩阵反推宏平均 P/R/F1，非训练日志直接打印值",
    }


def parse_ner_samples(limit: int = 3) -> List[Dict[str, str]]:
    ner_path = P1_DIR / "output" / "ner_hanlp.txt"
    txt = read_text_auto(ner_path)
    if not txt:
        return []

    blocks = [b.strip() for b in txt.split("\n\n") if b.strip()]
    with_entities: List[Dict[str, str]] = []
    without_entities: List[Dict[str, str]] = []
    for block in blocks:
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        if len(lines) < 4:
            continue
        sentence = lines[0]
        person = lines[1].replace("人名:", "").strip()
        location = lines[2].replace("地名:", "").strip()
        org = lines[3].replace("机构名:", "").strip()
        if person == "无":
            person = ""
        if location == "无":
            location = ""
        if org == "无":
            org = ""
        row = {
            "句子": sentence,
            "人名": person or "无",
            "地名": location or "无",
            "机构名": org or "无",
        }
        has_entity = any([person, location, org])
        if has_entity:
            with_entities.append(row)
        else:
            without_entities.append(row)

    rows: List[Dict[str, str]] = []
    rows.extend(with_entities[:limit])
    if len(rows) < limit:
        rows.extend(without_entities[: max(0, limit - len(rows))])
    return rows


def write_csv(rows: List[Dict[str, str]], csv_path: Path) -> None:
    fields = [
        "实验链路",
        "标签准确率",
        "标签级Macro Precision",
        "标签级Macro Recall",
        "标签级Macro F1",
        "token数",
        "PER实体数",
        "LOC实体数",
        "ORG实体数",
        "交叉验证Mean",
        "交叉验证Std",
        "训练样本数",
        "测试样本数",
        "指标来源",
        "备注",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_md(rows: List[Dict[str, str]], ner_samples: List[Dict[str, str]]) -> str:
    by_chain = {row["实验链路"]: row for row in rows}

    hmm_main = by_chain.get("P1 主线 HMM", {})
    bilstm = by_chain.get("P1/BiLSTMCRF 深度学习版本", {})
    hmm_test2 = parse_hmm_test2()

    lines: List[str] = []
    lines.append("# 统一实验结果汇总")
    lines.append("")
    lines.append("## 1. 实验链路说明")
    lines.append("- P1 主线 HMM：标准预处理、BMES 构建、HMM 训练与评估链路。")
    lines.append("- P1/BiLSTMCRF 深度学习版本：BiLSTM-CRF 训练评估链路。")
    lines.append("- P1/test2 迭代版 HMM：早期迭代探索与词典回灌备份，不作为主结果链路。")
    lines.append("")

    lines.append("## 2. 统一指标总表")
    lines.append("| 实验链路 | 标签准确率 | 标签级Macro Precision | 标签级Macro Recall | 标签级Macro F1 | token数 | PER实体数 | LOC实体数 | ORG实体数 | 交叉验证Mean | 交叉验证Std | 训练样本数 | 测试样本数 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['实验链路']} | {row['标签准确率']} | {row['标签级Macro Precision']} | {row['标签级Macro Recall']} | {row['标签级Macro F1']} | {row['token数']} | {row['PER实体数']} | {row['LOC实体数']} | {row['ORG实体数']} | {row['交叉验证Mean']} | {row['交叉验证Std']} | {row['训练样本数']} | {row['测试样本数']} |"
        )
    lines.append("")

    lines.append("## 2.1 数据口径说明")
    lines.append("- P1 主线 HMM 与 BiLSTM-CRF 的训练/测试样本数不同，当前结果用于方法趋势对照，不作为严格同测试集公平排名。")
    lines.append("- 若需要严格公平比较，应统一 train/dev/test 划分后重新训练。")
    lines.append("")

    lines.append("## 3. HMM 主线结果")
    lines.append(f"- 标签准确率: {hmm_main.get('标签准确率', '待补充')}")
    lines.append(f"- 标签级 Macro Precision/Recall/F1: {hmm_main.get('标签级Macro Precision', '待补充')} / {hmm_main.get('标签级Macro Recall', '待补充')} / {hmm_main.get('标签级Macro F1', '待补充')}")
    lines.append(f"- 交叉验证 Mean/Std: {hmm_main.get('交叉验证Mean', '待补充')} / {hmm_main.get('交叉验证Std', '待补充')}")
    lines.append(f"- 指标来源: {hmm_main.get('指标来源', '待补充')}")
    lines.append(f"- 备注: {hmm_main.get('备注', '待补充')}")
    lines.append("")

    lines.append("## 4. BiLSTM-CRF 结果")
    lines.append(f"- 标签准确率: {bilstm.get('标签准确率', '待补充')}")
    lines.append(f"- 标签级 Macro Precision/Recall/F1: {bilstm.get('标签级Macro Precision', '待补充')} / {bilstm.get('标签级Macro Recall', '待补充')} / {bilstm.get('标签级Macro F1', '待补充')}")
    lines.append(f"- 训练/测试样本数: {bilstm.get('训练样本数', '待补充')} / {bilstm.get('测试样本数', '待补充')}")
    lines.append(f"- 指标来源: {bilstm.get('指标来源', '待补充')}")
    lines.append(f"- 备注: {bilstm.get('备注', '待补充')}")
    lines.append("")

    nlp4j = by_chain.get("P1/nlp4j_baseline NLP4J 对照实验", {})
    nlp4j_hmm = by_chain.get("P1/nlp4j_baseline Java HMM 训练式基线", {})
    lines.append("## 5. NLP4J 对照实验")
    lines.append(f"- 当前状态: {nlp4j.get('备注', '待补充')}")
    lines.append(f"- 转换结果: {nlp4j.get('token数', '待补充')} 个 token, PER {nlp4j.get('PER实体数', '待补充')} 个, LOC {nlp4j.get('LOC实体数', '待补充')} 个, ORG {nlp4j.get('ORG实体数', '待补充')} 个")
    lines.append(f"- Accuracy/Precision/Recall/F1: {nlp4j.get('标签准确率', '待补充')} / {nlp4j.get('标签级Macro Precision', '待补充')} / {nlp4j.get('标签级Macro Recall', '待补充')} / {nlp4j.get('标签级Macro F1', '待补充')}")
    lines.append(f"- 指标来源: {nlp4j.get('指标来源', '待补充')}")
    lines.append("- 说明: 当前仅做转换与占位汇总，不把 sample_output.txt 当作真实实验指标。")
    if nlp4j_hmm:
        lines.append(f"- 训练式 HMM: {nlp4j_hmm.get('标签准确率', '待补充')} / {nlp4j_hmm.get('标签级Macro Precision', '待补充')} / {nlp4j_hmm.get('标签级Macro Recall', '待补充')} / {nlp4j_hmm.get('标签级Macro F1', '待补充')}")
        lines.append(f"- 训练式 HMM 指标来源: {nlp4j_hmm.get('指标来源', '待补充')}")
    lines.append("")

    lines.append("## 6. NER 结果样例")
    if not ner_samples:
        lines.append("- 待补充（未解析到 P1/output/ner_hanlp.txt 样例）。")
    else:
        has_entity_sample = any(
            (item.get("人名") != "无" or item.get("地名") != "无" or item.get("机构名") != "无")
            for item in ner_samples
        )
        if not has_entity_sample:
            lines.append("- 当前测试样例未识别出明显 PER/LOC/ORG。")
        for i, item in enumerate(ner_samples, start=1):
            lines.append(f"### 样例 {i}")
            lines.append(f"- {item['句子']}")
            lines.append(f"- 人名: {item['人名']}")
            lines.append(f"- 地名: {item['地名']}")
            lines.append(f"- 机构名: {item['机构名']}")
            lines.append("")

    lines.append("## 7. 当前不足")
    lines.append("- BiLSTM-CRF 当前主要从混淆矩阵反推指标，缺少统一文本日志沉淀。")
    lines.append("- 三条链路的指标命名和落盘格式仍未完全统一。")
    lines.append("")

    lines.append("## 8. 下一步需要补 NLP4J")
    lines.append("- 若后续接入真实 NLP4J 运行环境，可将 Accuracy/F1 纳入同一份 metrics_summary.csv。")
    lines.append("- 也可补充标准答案文件，将转换结果与真实指标一起评估。")
    lines.append("")

    lines.append("## 9. 附录：test2 迭代探索结果")
    if hmm_test2.get("has_metrics"):
        m_metrics = parse_label_report_m_metrics(hmm_test2.get("标签报告路径", ""))
        lines.append(f"- 标签准确率: {hmm_test2.get('标签准确率', '待补充')}")
        lines.append(f"- 宏平均 F1: {hmm_test2.get('标签级Macro F1', '待补充')}")
        lines.append(f"- M 标签 Recall/F1: {m_metrics.get('m_recall', '待补充')} / {m_metrics.get('m_f1', '待补充')}")
        lines.append(f"- 混淆矩阵: {hmm_test2.get('混淆矩阵路径', '待补充')}")
        lines.append(f"- 标签报告: {hmm_test2.get('标签报告路径', '待补充')}")
        lines.append("- 结论: test2 作为词典回灌与错误分析探索版本保留，不作为最终主模型结果。")
    else:
        lines.append("- 当前未生成结构化评估输出，暂无法汇总 test2 指标。")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        parse_hmm_main(),
        parse_bilstm_crf(),
        parse_nlp4j_baseline(),
        parse_nlp4j_hmm(),
    ]
    ner_samples = parse_ner_samples(limit=3)

    write_csv(rows, SUMMARY_CSV)
    SUMMARY_MD.write_text(build_md(rows, ner_samples), encoding="utf-8")

    print(f"Saved: {SUMMARY_MD}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
