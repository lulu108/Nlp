# -*- coding: utf-8 -*-
"""
功能：
1. 读取 train.txt / test.txt
2. 统计 HMM 参数：
   - 初始状态概率
   - 状态转移概率
   - 发射概率
3. 使用 Viterbi 对测试集预测
4. 计算标签级准确率
5. 输出部分预测结果
"""

from pathlib import Path
from collections import defaultdict
import json
import math
from typing import Dict, List, Tuple

TRAIN_PATH = Path("./P1/test2/datasets/auto/train.txt")
TEST_PATH = Path("./P1/test2/datasets/auto/test.txt")
OUTPUT_DIR = Path("./P1/test2/output")
METRICS_JSON = OUTPUT_DIR / "metrics.json"
METRICS_SUMMARY = OUTPUT_DIR / "metrics_summary.txt"
CONFUSION_TSV = OUTPUT_DIR / "confusion_matrix.tsv"
CONFUSION_PNG = OUTPUT_DIR / "confusion_matrix.png"
LABEL_REPORT_TSV = OUTPUT_DIR / "label_report.tsv"

STATES = ["B", "M", "E", "S"]


def load_dataset(path: Path):
    """
    读取格式：
    原句 \t 字序列 \t 标签序列 \t 分词结果
    """
    samples = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 4:
            continue
        sentence = parts[0]
        chars = parts[1].split()
        tags = parts[2].split()
        words = parts[3].split()

        if len(chars) != len(tags):
            continue

        samples.append({
            "sentence": sentence,
            "chars": chars,
            "tags": tags,
            "words": words
        })
    return samples


class HMM:
    def __init__(self):
        self.states = STATES

        # 计数
        self.pi_count = defaultdict(int)                        # 初始状态计数
        self.trans_count = {s: defaultdict(int) for s in STATES}  # 转移计数
        self.emit_count = {s: defaultdict(int) for s in STATES}   # 发射计数
        self.state_count = defaultdict(int)                    # 状态总计数

        # 概率（对数形式）
        self.pi = {}
        self.trans = {s: {} for s in STATES}
        self.emit = {s: {} for s in STATES}

        self.vocab = set()

    def train(self, samples):
        for sample in samples:
            chars = sample["chars"]
            tags = sample["tags"]

            if not chars or not tags:
                continue

            # 初始状态
            self.pi_count[tags[0]] += 1

            for i, (ch, tag) in enumerate(zip(chars, tags)):
                self.state_count[tag] += 1
                self.emit_count[tag][ch] += 1
                self.vocab.add(ch)

                if i > 0:
                    prev_tag = tags[i - 1]
                    self.trans_count[prev_tag][tag] += 1

        self._calc_probabilities()

    def _calc_probabilities(self):
        """
        使用加1平滑，并转换为对数概率，避免下溢
        """
        num_states = len(self.states)
        vocab_size = len(self.vocab)

        total_sentences = sum(self.pi_count.values())

        # 初始概率
        for s in self.states:
            prob = (self.pi_count[s] + 1) / (total_sentences + num_states)
            self.pi[s] = math.log(prob)

        # 转移概率
        for s1 in self.states:
            total = sum(self.trans_count[s1].values())
            for s2 in self.states:
                prob = (self.trans_count[s1][s2] + 1) / (total + num_states)
                self.trans[s1][s2] = math.log(prob)

        # 发射概率
        for s in self.states:
            total = self.state_count[s]
            for ch in self.vocab:
                prob = (self.emit_count[s][ch] + 1) / (total + vocab_size)
                self.emit[s][ch] = math.log(prob)

        self.unknown_emit = {}
        for s in self.states:
            total = self.state_count[s]
            prob = 1 / (total + vocab_size)
            self.unknown_emit[s] = math.log(prob)

    def get_emit_logprob(self, state, ch):
        return self.emit[state].get(ch, self.unknown_emit[state])

    def viterbi(self, chars):
        """
        给定字序列，返回最优标签序列
        """
        V = [{}]
        path = {}

        # 初始化
        for s in self.states:
            V[0][s] = self.pi[s] + self.get_emit_logprob(s, chars[0])
            path[s] = [s]

        # 递推
        for t in range(1, len(chars)):
            V.append({})
            new_path = {}

            for curr_state in self.states:
                candidates = []
                for prev_state in self.states:
                    score = (
                        V[t - 1][prev_state]
                        + self.trans[prev_state][curr_state]
                        + self.get_emit_logprob(curr_state, chars[t])
                    )
                    candidates.append((score, prev_state))

                best_score, best_prev = max(candidates, key=lambda x: x[0])
                V[t][curr_state] = best_score
                new_path[curr_state] = path[best_prev] + [curr_state]

            path = new_path

        # 终止
        final_score, final_state = max((V[-1][s], s) for s in self.states)
        return path[final_state]

    @staticmethod
    def tags_to_words(chars, tags):
        """
        根据 BMES 标签恢复分词结果
        """
        words = []
        buffer = ""

        for ch, tag in zip(chars, tags):
            if tag == "S":
                if buffer:
                    words.append(buffer)
                    buffer = ""
                words.append(ch)
            elif tag == "B":
                if buffer:
                    words.append(buffer)
                buffer = ch
            elif tag == "M":
                buffer += ch
            elif tag == "E":
                buffer += ch
                words.append(buffer)
                buffer = ""
            else:
                # 异常保护
                if buffer:
                    words.append(buffer)
                    buffer = ""
                words.append(ch)

        if buffer:
            words.append(buffer)

        return words


def evaluate(model: HMM, samples):
    total_tags = 0
    correct_tags = 0
    label_to_idx = {label: idx for idx, label in enumerate(STATES)}
    confusion = [[0 for _ in STATES] for _ in STATES]

    show_results = []

    for sample in samples:
        chars = sample["chars"]
        true_tags = sample["tags"]

        pred_tags = model.viterbi(chars)

        for t1, t2 in zip(true_tags, pred_tags):
            total_tags += 1
            if t1 == t2:
                correct_tags += 1
            if t1 in label_to_idx and t2 in label_to_idx:
                confusion[label_to_idx[t1]][label_to_idx[t2]] += 1

        if len(show_results) < 5:
            pred_words = model.tags_to_words(chars, pred_tags)
            show_results.append({
                "sentence": sample["sentence"],
                "true_words": sample["words"],
                "pred_words": pred_words,
                "true_tags": true_tags,
                "pred_tags": pred_tags
            })

    acc = correct_tags / total_tags if total_tags > 0 else 0
    label_report, macro_p, macro_r, macro_f1 = build_label_report(confusion)
    return acc, show_results, confusion, label_report, macro_p, macro_r, macro_f1


def build_label_report(confusion: List[List[int]]) -> Tuple[List[Dict[str, float]], float, float, float]:
    report: List[Dict[str, float]] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for idx, label in enumerate(STATES):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(len(STATES))) - tp
        fn = sum(confusion[idx][col] for col in range(len(STATES))) - tp
        support = sum(confusion[idx])

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        report.append({
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_p = sum(precisions) / len(precisions) if precisions else 0.0
    macro_r = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return report, macro_p, macro_r, macro_f1


def write_confusion_tsv(confusion: List[List[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["true\\pred\t" + "\t".join(STATES)]
    for label, row in zip(STATES, confusion):
        lines.append(label + "\t" + "\t".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_label_report_tsv(report: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "label\tprecision\trecall\tf1\tsupport"
    lines = [header]
    for row in report:
        lines.append(
            f"{row['label']}\t{row['precision']:.6f}\t{row['recall']:.6f}\t{row['f1']:.6f}\t{int(row['support'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confusion_png(confusion: List[List[int]], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Matplotlib unavailable, skip confusion_matrix.png: {exc}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(STATES)), STATES)
    ax.set_yticks(range(len(STATES)), STATES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("P1/test2 HMM BMES Confusion Matrix")

    for i in range(len(STATES)):
        for j in range(len(STATES)):
            ax.text(j, i, str(confusion[i][j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    train_samples = load_dataset(TRAIN_PATH)
    test_samples = load_dataset(TEST_PATH)

    print(f"训练集样本数: {len(train_samples)}")
    print(f"测试集样本数: {len(test_samples)}")

    model = HMM()
    model.train(train_samples)

    print("\n===== HMM 参数示例 =====")
    print("初始状态概率（log）:")
    for s in STATES:
        print(f"{s}: {model.pi[s]:.6f}")

    print("\n转移概率（log）示例:")
    for s1 in STATES:
        row = {s2: round(model.trans[s1][s2], 4) for s2 in STATES}
        print(f"{s1} -> {row}")

    acc, examples, confusion, label_report, macro_p, macro_r, macro_f1 = evaluate(model, test_samples)
    print(f"\n测试集标签级准确率: {acc:.4f}")

    print("\n===== 预测样例 =====")
    for i, ex in enumerate(examples, 1):
        print(f"\n样例 {i}")
        print("原句:", ex["sentence"])
        print("真实分词:", "/".join(ex["true_words"]))
        print("预测分词:", "/".join(ex["pred_words"]))
        print("真实标签:", " ".join(ex["true_tags"]))
        print("预测标签:", " ".join(ex["pred_tags"]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_confusion_tsv(confusion, CONFUSION_TSV)
    write_confusion_png(confusion, CONFUSION_PNG)
    write_label_report_tsv(label_report, LABEL_REPORT_TSV)

    metrics = {
        "chain": "P1/test2 迭代版 HMM",
        "tag_accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "source": "P1/test2/03train_hmm.py",
        "confusion_matrix_path": str(CONFUSION_TSV),
        "label_report_path": str(LABEL_REPORT_TSV),
    }
    METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        f"标签准确率: {acc:.6f}",
        f"Macro Precision: {macro_p:.6f}",
        f"Macro Recall: {macro_r:.6f}",
        f"Macro F1: {macro_f1:.6f}",
        f"训练样本数: {len(train_samples)}",
        f"测试样本数: {len(test_samples)}",
        f"混淆矩阵文件路径: {CONFUSION_TSV}",
        f"标签报告文件路径: {LABEL_REPORT_TSV}",
    ]
    METRICS_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()