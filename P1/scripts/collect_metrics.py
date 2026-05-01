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
            "词级Precision": fmt_metric(macro_p),
            "词级Recall": fmt_metric(macro_r),
            "词级F1": fmt_metric(macro_f1),
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
        "词级Precision": fmt_metric(macro_p),
        "词级Recall": fmt_metric(macro_r),
        "词级F1": fmt_metric(macro_f1),
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
    train_count = count_non_empty_lines(P1_DIR / "test2" / "datasets" / "auto" / "train.txt")
    test_count = count_non_empty_lines(P1_DIR / "test2" / "datasets" / "auto" / "test.txt")

    return {
        "实验链路": "P1/test2 迭代版 HMM",
        "标签准确率": "待补充",
        "词级Precision": "待补充",
        "词级Recall": "待补充",
        "词级F1": "待补充",
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
    }


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
    note = "已检测到转换结果；真实 Accuracy/Precision/Recall/F1 仍待补充" if has_result else "尚未生成转换结果；真实指标待补充"

    return {
        "实验链路": "P1/nlp4j_baseline NLP4J 对照实验",
        "标签准确率": "待补充",
        "词级Precision": "待补充",
        "词级Recall": "待补充",
        "词级F1": "待补充",
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


def parse_bilstm_crf() -> Dict[str, str]:
    confusion_path = P1_DIR / "BiLSTMCRF" / "output" / "confusion_matrix.tsv"
    confusion = parse_confusion_tsv(confusion_path)

    train_count = count_non_empty_lines(P1_DIR / "BiLSTMCRF" / "data" / "train.txt")
    test_count = count_non_empty_lines(P1_DIR / "BiLSTMCRF" / "data" / "test.txt")

    if confusion is None:
        return {
            "实验链路": "P1/BiLSTMCRF 深度学习版本",
            "标签准确率": "待补充",
            "词级Precision": "待补充",
            "词级Recall": "待补充",
            "词级F1": "待补充",
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
        "词级Precision": fmt_metric(p),
        "词级Recall": fmt_metric(r),
        "词级F1": fmt_metric(f1),
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
    rows: List[Dict[str, str]] = []
    for block in blocks:
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        if len(lines) < 4:
            continue
        sentence = lines[0]
        person = lines[1].replace("人名:", "").strip()
        location = lines[2].replace("地名:", "").strip()
        org = lines[3].replace("机构名:", "").strip()
        rows.append({
            "句子": sentence,
            "人名": person or "无",
            "地名": location or "无",
            "机构名": org or "无",
        })
        if len(rows) >= limit:
            break
    return rows


def write_csv(rows: List[Dict[str, str]], csv_path: Path) -> None:
    fields = [
        "实验链路",
        "标签准确率",
        "词级Precision",
        "词级Recall",
        "词级F1",
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
            writer.writerow(row)


def build_md(rows: List[Dict[str, str]], ner_samples: List[Dict[str, str]]) -> str:
    by_chain = {row["实验链路"]: row for row in rows}

    hmm_main = by_chain.get("P1 主线 HMM", {})
    hmm_test2 = by_chain.get("P1/test2 迭代版 HMM", {})
    bilstm = by_chain.get("P1/BiLSTMCRF 深度学习版本", {})

    lines: List[str] = []
    lines.append("# 统一实验结果汇总")
    lines.append("")
    lines.append("## 1. 实验链路说明")
    lines.append("- P1 主线 HMM：标准预处理、BMES 构建、HMM 训练与评估链路。")
    lines.append("- P1/test2 迭代版 HMM：面向词典回灌和误差修复的迭代链路。")
    lines.append("- P1/BiLSTMCRF 深度学习版本：BiLSTM-CRF 训练评估链路。")
    lines.append("")

    lines.append("## 2. 统一指标总表")
    lines.append("| 实验链路 | 标签准确率 | 词级Precision | 词级Recall | 词级F1 | token数 | PER实体数 | LOC实体数 | ORG实体数 | 交叉验证Mean | 交叉验证Std | 训练样本数 | 测试样本数 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['实验链路']} | {row['标签准确率']} | {row['词级Precision']} | {row['词级Recall']} | {row['词级F1']} | {row['token数']} | {row['PER实体数']} | {row['LOC实体数']} | {row['ORG实体数']} | {row['交叉验证Mean']} | {row['交叉验证Std']} | {row['训练样本数']} | {row['测试样本数']} |"
        )
    lines.append("")

    lines.append("## 3. HMM 主线结果")
    lines.append(f"- 标签准确率: {hmm_main.get('标签准确率', '待补充')}")
    lines.append(f"- 词级 Precision/Recall/F1: {hmm_main.get('词级Precision', '待补充')} / {hmm_main.get('词级Recall', '待补充')} / {hmm_main.get('词级F1', '待补充')}")
    lines.append(f"- 交叉验证 Mean/Std: {hmm_main.get('交叉验证Mean', '待补充')} / {hmm_main.get('交叉验证Std', '待补充')}")
    lines.append(f"- 指标来源: {hmm_main.get('指标来源', '待补充')}")
    lines.append(f"- 备注: {hmm_main.get('备注', '待补充')}")
    lines.append("")

    lines.append("## 4. test2 迭代链路结果")
    lines.append(f"- 标签准确率: {hmm_test2.get('标签准确率', '待补充')}")
    lines.append(f"- 词级 Precision/Recall/F1: {hmm_test2.get('词级Precision', '待补充')} / {hmm_test2.get('词级Recall', '待补充')} / {hmm_test2.get('词级F1', '待补充')}")
    lines.append(f"- 训练/测试样本数: {hmm_test2.get('训练样本数', '待补充')} / {hmm_test2.get('测试样本数', '待补充')}")
    lines.append(f"- 指标来源: {hmm_test2.get('指标来源', '待补充')}")
    lines.append(f"- 备注: {hmm_test2.get('备注', '待补充')}")
    lines.append("")

    lines.append("## 5. BiLSTM-CRF 结果")
    lines.append(f"- 标签准确率: {bilstm.get('标签准确率', '待补充')}")
    lines.append(f"- 词级 Precision/Recall/F1: {bilstm.get('词级Precision', '待补充')} / {bilstm.get('词级Recall', '待补充')} / {bilstm.get('词级F1', '待补充')}")
    lines.append(f"- 训练/测试样本数: {bilstm.get('训练样本数', '待补充')} / {bilstm.get('测试样本数', '待补充')}")
    lines.append(f"- 指标来源: {bilstm.get('指标来源', '待补充')}")
    lines.append(f"- 备注: {bilstm.get('备注', '待补充')}")
    lines.append("")

    nlp4j = by_chain.get("P1/nlp4j_baseline NLP4J 对照实验", {})
    lines.append("## 6. NLP4J 对照实验")
    lines.append(f"- 当前状态: {nlp4j.get('备注', '待补充')}")
    lines.append(f"- 转换结果: {nlp4j.get('token数', '待补充')} 个 token, PER {nlp4j.get('PER实体数', '待补充')} 个, LOC {nlp4j.get('LOC实体数', '待补充')} 个, ORG {nlp4j.get('ORG实体数', '待补充')} 个")
    lines.append(f"- Accuracy/Precision/Recall/F1: {nlp4j.get('标签准确率', '待补充')} / {nlp4j.get('词级Precision', '待补充')} / {nlp4j.get('词级Recall', '待补充')} / {nlp4j.get('词级F1', '待补充')}")
    lines.append(f"- 指标来源: {nlp4j.get('指标来源', '待补充')}")
    lines.append("- 说明: 当前仅做转换与占位汇总，不把 sample_output.txt 当作真实实验指标。")
    lines.append("")

    lines.append("## 7. NER 结果样例")
    if not ner_samples:
        lines.append("- 待补充（未解析到 P1/output/ner_hanlp.txt 样例）。")
    else:
        for i, item in enumerate(ner_samples, start=1):
            lines.append(f"### 样例 {i}")
            lines.append(f"- {item['句子']}")
            lines.append(f"- 人名: {item['人名']}")
            lines.append(f"- 地名: {item['地名']}")
            lines.append(f"- 机构名: {item['机构名']}")
            lines.append("")

    lines.append("## 8. 当前不足")
    lines.append("- test2 当前缺少结构化评测日志，准确率与 F1 暂无法自动汇总。")
    lines.append("- BiLSTM-CRF 当前主要从混淆矩阵反推指标，缺少统一文本日志沉淀。")
    lines.append("- 三条链路的指标命名和落盘格式仍未完全统一。")
    lines.append("")

    lines.append("## 9. 下一步需要补 NLP4J")
    lines.append("- 若后续接入真实 NLP4J 运行环境，可将 Accuracy/F1 纳入同一份 metrics_summary.csv。")
    lines.append("- 也可补充标准答案文件，将转换结果与真实指标一起评估。")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        parse_hmm_main(),
        parse_hmm_test2(),
        parse_bilstm_crf(),
        parse_nlp4j_baseline(),
    ]
    ner_samples = parse_ner_samples(limit=3)

    write_csv(rows, SUMMARY_CSV)
    SUMMARY_MD.write_text(build_md(rows, ner_samples), encoding="utf-8")

    print(f"Saved: {SUMMARY_MD}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
