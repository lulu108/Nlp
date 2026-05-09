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

原始HMM主线：python P1/03train_hmm.py --dataset auto > P1/logs/hmm_main_latest.txt
使用lstm数据集：python P1/03train_hmm.py --dataset bilstmcrf > P1/logs/hmm_bilstm_data_latest.txt
"""

from pathlib import Path
from collections import defaultdict
import argparse
import math
import random
import csv

AUTO_TRAIN_PATH = Path("./P1/datasets/auto/train.txt")
AUTO_TEST_PATH = Path("./P1/datasets/auto/test.txt")
BITLSTMCRF_TRAIN_PATH = Path("./P1/BiLSTMCRF/data/train.txt")
BITLSTMCRF_TEST_PATH = Path("./P1/BiLSTMCRF/data/test.txt")
REPORT_DIR = Path("./P1/reports")
HMM_DEV_TUNE_CSV = REPORT_DIR / "hmm_dev_tune.csv"
HMM_DEV_TUNE_MD = REPORT_DIR / "hmm_dev_tune.md"

STATES = ["B", "M", "E", "S"]
ALPHA_CANDIDATES = [1.0, 0.5, 0.1, 0.05, 0.01]


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


def resolve_dataset(dataset_name: str):
    if dataset_name == "auto":
        return dataset_name, AUTO_TRAIN_PATH, AUTO_TEST_PATH, None
    if dataset_name == "bilstmcrf":
        missing = []
        if not BITLSTMCRF_TRAIN_PATH.exists():
            missing.append(BITLSTMCRF_TRAIN_PATH)
        if not BITLSTMCRF_TEST_PATH.exists():
            missing.append(BITLSTMCRF_TEST_PATH)
        if missing:
            message = [
                "选择 bilstmcrf 数据集时，以下文件不存在：",
                *[f"- {path}" for path in missing],
                "请先运行：",
                "python P1/BiLSTMCRF/01build_bmes_from_raw.py",
                "python P1/BiLSTMCRF/02_prepare_data.py",
            ]
            raise FileNotFoundError("\n".join(message))
        return dataset_name, BITLSTMCRF_TRAIN_PATH, BITLSTMCRF_TEST_PATH, None
    raise ValueError(f"不支持的数据集类型: {dataset_name}")


def parse_alpha_candidates(text: str):
    raw = [x.strip() for x in text.split(",") if x.strip()]
    if not raw:
        raise ValueError("alpha candidates 不能为空")
    values = []
    for item in raw:
        try:
            alpha = float(item)
        except Exception as exc:
            raise ValueError(f"alpha candidates 解析失败: {item}") from exc
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        values.append(alpha)
    return values


def split_train_dev(train_samples, dev_ratio: float, seed: int):
    if not (0.0 < dev_ratio < 1.0):
        raise ValueError("dev_ratio must be between 0 and 1")

    if len(train_samples) < 2:
        raise ValueError("训练样本过少，无法执行 train/dev 划分")

    shuffled = list(train_samples)
    random.Random(seed).shuffle(shuffled)

    dev_size = max(1, int(len(shuffled) * dev_ratio))
    if dev_size >= len(shuffled):
        dev_size = len(shuffled) - 1

    dev_samples = shuffled[:dev_size]
    tune_train_samples = shuffled[dev_size:]
    return tune_train_samples, dev_samples


class HMM:
    def __init__(self, alpha: float = 1.0, use_bmes_constraints: bool = True):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.states = STATES
        self.use_bmes_constraints = use_bmes_constraints

        # BMES legality constraints
        self.ALLOWED_START = {"B", "S"}
        self.ALLOWED_END = {"E", "S"}
        self.ALLOWED_TRANSITIONS = {
            "B": {"M", "E"},
            "M": {"M", "E"},
            "E": {"B", "S"},
            "S": {"B", "S"},
        }

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
        self.alpha = alpha

    def train(self, samples, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
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
        使用 alpha 平滑，并转换为对数概率，避免下溢
        """
        num_states = len(self.states)
        vocab_size = len(self.vocab)
        alpha = self.alpha

        total_sentences = sum(self.pi_count.values())

        # 初始概率
        for s in self.states:
            prob = (self.pi_count[s] + alpha) / (total_sentences + alpha * num_states)
            self.pi[s] = math.log(prob)

        # 转移概率
        for s1 in self.states:
            total = sum(self.trans_count[s1].values())
            for s2 in self.states:
                prob = (self.trans_count[s1][s2] + alpha) / (total + alpha * num_states)
                self.trans[s1][s2] = math.log(prob)

        # 发射概率
        for s in self.states:
            total = self.state_count[s]
            for ch in self.vocab:
                prob = (self.emit_count[s][ch] + alpha) / (total + alpha * vocab_size)
                self.emit[s][ch] = math.log(prob)

        self.unknown_emit = {}
        for s in self.states:
            total = self.state_count[s]
            prob = alpha / (total + alpha * vocab_size)
            self.unknown_emit[s] = math.log(prob)

    def get_emit_logprob(self, state, ch):
        return self.emit[state].get(ch, self.unknown_emit[state])

    def viterbi(self, chars):
        """
        给定字序列，返回最优标签序列
        """
        neg_inf = -1e18
        V = [{}]
        path = {}
        use_constraints = self.use_bmes_constraints

        # 初始化
        for s in self.states:
            if use_constraints and s not in self.ALLOWED_START:
                V[0][s] = neg_inf
                path[s] = [s]
                continue
            V[0][s] = self.pi[s] + self.get_emit_logprob(s, chars[0])
            path[s] = [s]

        # 递推
        for t in range(1, len(chars)):
            V.append({})
            new_path = {}

            for curr_state in self.states:
                candidates = []
                for prev_state in self.states:
                    if use_constraints and curr_state not in self.ALLOWED_TRANSITIONS.get(prev_state, set()):
                        continue
                    score = (
                        V[t - 1][prev_state]
                        + self.trans[prev_state][curr_state]
                        + self.get_emit_logprob(curr_state, chars[t])
                    )
                    candidates.append((score, prev_state))
                if candidates:
                    best_score, best_prev = max(candidates, key=lambda x: x[0])
                    V[t][curr_state] = best_score
                    new_path[curr_state] = path[best_prev] + [curr_state]
                else:
                    V[t][curr_state] = neg_inf
                    new_path[curr_state] = path.get(curr_state, [curr_state]) + [curr_state]

            path = new_path

        # 终止
        if use_constraints:
            final_candidates = [(V[-1][s], s) for s in self.states if s in self.ALLOWED_END]
        else:
            final_candidates = [(V[-1][s], s) for s in self.states]
        final_score, final_state = max(final_candidates, key=lambda x: x[0])
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
    state_index = {s: i for i, s in enumerate(STATES)}
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
            confusion[state_index[t1]][state_index[t2]] += 1

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
    return acc, show_results, confusion


def evaluate_split(model: HMM, samples):
    acc, _, confusion = evaluate(model, samples)
    metrics = calc_precision_recall_f1(confusion, STATES)
    macro_precision = sum(metrics[s][0] for s in STATES) / len(STATES)
    macro_recall = sum(metrics[s][1] for s in STATES) / len(STATES)
    macro_f1 = sum(metrics[s][2] for s in STATES) / len(STATES)
    return acc, macro_precision, macro_recall, macro_f1, confusion


def train_model(train_samples, alpha: float, use_bmes_constraints: bool) -> HMM:
    model = HMM(alpha=alpha, use_bmes_constraints=use_bmes_constraints)
    model.train(train_samples, alpha=alpha)
    return model


def tune_alpha(train_samples, dev_samples, alpha_candidates, use_bmes_constraints: bool):
    results = []
    best_alpha = None
    best_macro_f1 = -1.0
    best_acc = -1.0

    for alpha in alpha_candidates:
        model = train_model(train_samples, alpha, use_bmes_constraints)
        acc, macro_precision, macro_recall, macro_f1, _ = evaluate_split(model, dev_samples)
        result = {
            "alpha": alpha,
            "dev_accuracy": acc,
            "dev_macro_precision": macro_precision,
            "dev_macro_recall": macro_recall,
            "dev_macro_f1": macro_f1,
        }
        results.append(result)

        # Tie-break: macro_f1 更高优先；若相同，accuracy 更高优先；再相同，较小 alpha 优先。
        if (
            macro_f1 > best_macro_f1
            or (abs(macro_f1 - best_macro_f1) < 1e-12 and acc > best_acc)
            or (
                abs(macro_f1 - best_macro_f1) < 1e-12
                and abs(acc - best_acc) < 1e-12
                and (best_alpha is None or alpha < best_alpha)
            )
        ):
            best_macro_f1 = macro_f1
            best_acc = acc
            best_alpha = alpha

    return best_alpha, results


def save_dev_tune_reports(results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with HMM_DEV_TUNE_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "dev_accuracy",
                "dev_macro_precision",
                "dev_macro_recall",
                "dev_macro_f1",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({
                "alpha": f"{row['alpha']:.6f}",
                "dev_accuracy": f"{row['dev_accuracy']:.6f}",
                "dev_macro_precision": f"{row['dev_macro_precision']:.6f}",
                "dev_macro_recall": f"{row['dev_macro_recall']:.6f}",
                "dev_macro_f1": f"{row['dev_macro_f1']:.6f}",
            })

    lines = [
        "| alpha | Dev Accuracy | Dev Macro Precision | Dev Macro Recall | Dev Macro F1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| "
            f"{row['alpha']:.6f} | {row['dev_accuracy']:.4f} | "
            f"{row['dev_macro_precision']:.4f} | {row['dev_macro_recall']:.4f} | {row['dev_macro_f1']:.4f} |"
        )
    HMM_DEV_TUNE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_confusion_matrix(confusion, states):
    header = "true\\pred\t" + "\t".join(states)
    print(header)
    for i, s in enumerate(states):
        row = "\t".join(str(confusion[i][j]) for j in range(len(states)))
        print(f"{s}\t{row}")


def calc_precision_recall_f1(confusion, states):
    metrics = {}
    for i, s in enumerate(states):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(len(states))) - tp
        fn = sum(confusion[i][c] for c in range(len(states))) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[s] = (precision, recall, f1)
    return metrics


def save_confusion_matrix_plot(confusion, states, out_path: Path):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(confusion, cmap="Blues")

        ax.set_xticks(range(len(states)))
        ax.set_yticks(range(len(states)))
        ax.set_xticklabels(states)
        ax.set_yticklabels(states)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("BMES Confusion Matrix")

        # annotate cells
        for i in range(len(states)):
            for j in range(len(states)):
                ax.text(j, i, str(confusion[i][j]), ha="center", va="center", fontsize=8)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        print("混淆矩阵图保存失败，已跳过。")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate BMES HMM")
    parser.add_argument(
        "--dataset",
        choices=["auto", "bilstmcrf"],
        default="auto",
        help="数据来源：auto 使用 P1 主线划分；bilstmcrf 使用 BiLSTM-CRF 同划分",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="平滑系数")
    parser.add_argument(
        "--use-bmes-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在 Viterbi 中使用 BMES 合法约束",
    )
    parser.add_argument(
        "--use-dev-tune",
        action="store_true",
        help="从 train.txt 内部划分 dev 集并选择最优 alpha",
    )
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="train 内部划分 dev 的比例")
    parser.add_argument("--seed", type=int, default=42, help="train/dev 随机划分种子")
    parser.add_argument(
        "--alpha-candidates",
        type=str,
        default="1.0,0.5,0.1,0.05,0.01",
        help="自动调参时使用的 alpha 候选，逗号分隔",
    )
    parser.add_argument(
        "--train-with-dev",
        action="store_true",
        help="兼容旧参数：开启后行为与默认一致（最终均在完整 train.txt 上重训）",
    )
    args = parser.parse_args()

    if args.alpha <= 0:
        raise ValueError("alpha must be positive")

    dataset_name, train_path, test_path, dev_path = resolve_dataset(args.dataset)
    alpha_candidates = parse_alpha_candidates(args.alpha_candidates)

    print(f"数据源: {dataset_name}")
    print(f"训练集路径: {train_path}")
    print(f"测试集路径: {test_path}")
    if dev_path is not None:
        print(f"验证集路径: {dev_path}")
    print(f"BMES 合法约束: {'开启' if args.use_bmes_constraints else '关闭'}")
    print(f"初始 alpha: {args.alpha}")
    print(f"使用 dev 调参: {'开启' if args.use_dev_tune else '关闭'}")
    print(f"alpha candidates: {alpha_candidates}")
    print(f"dev_ratio: {args.dev_ratio}")
    print(f"seed: {args.seed}")
    print(f"使用 train+dev 训练最终模型: {'开启' if args.train_with_dev else '关闭'}")

    train_samples = load_dataset(train_path)
    test_samples = load_dataset(test_path)
    dev_samples = []

    print(f"训练集样本数: {len(train_samples)}")
    print(f"测试集样本数: {len(test_samples)}")

    selected_alpha = args.alpha
    if args.use_dev_tune:
        tune_train_samples, dev_samples = split_train_dev(train_samples, args.dev_ratio, args.seed)
        print("\n===== HMM Dev Tune =====")
        print(f"train samples for tune: {len(tune_train_samples)}")
        print(f"dev samples: {len(dev_samples)}")
        print(f"alpha candidates: {alpha_candidates}")

        best_alpha, tune_results = tune_alpha(
            tune_train_samples,
            dev_samples,
            alpha_candidates,
            args.use_bmes_constraints,
        )
        for row in tune_results:
            print(
                f"alpha={row['alpha']:<5.2f} "
                f"dev_acc={row['dev_accuracy']:.4f} "
                f"dev_macro_f1={row['dev_macro_f1']:.4f}"
            )

        save_dev_tune_reports(tune_results)
        selected_alpha = best_alpha if best_alpha is not None else args.alpha
        print(f"Best alpha: {selected_alpha:.2f}")
        print("Retrain on full train set...")

    final_train_samples = list(train_samples)
    print(f"最终训练集: full train (样本数={len(final_train_samples)})")

    print(f"最终使用 alpha: {selected_alpha:.2f}")

    model = train_model(final_train_samples, selected_alpha, args.use_bmes_constraints)

    print("\n===== HMM 参数示例 =====")
    print("初始状态概率（log）:")
    for s in STATES:
        print(f"{s}: {model.pi[s]:.6f}")

    print("\n转移概率（log）示例:")
    for s1 in STATES:
        row = {s2: round(model.trans[s1][s2], 4) for s2 in STATES}
        print(f"{s1} -> {row}")

    acc, examples, confusion = evaluate(model, test_samples)
    print(f"\n测试集标签级准确率: {acc:.4f}")

    print("\n===== Confusion Matrix (BMES) =====")
    print_confusion_matrix(confusion, STATES)

    print("\n===== Tag Precision / Recall / F1 =====")
    metrics = calc_precision_recall_f1(confusion, STATES)
    macro_precision = sum(metrics[s][0] for s in STATES) / len(STATES)
    macro_recall = sum(metrics[s][1] for s in STATES) / len(STATES)
    macro_f1 = sum(metrics[s][2] for s in STATES) / len(STATES)
    for s in STATES:
        p, r, f1 = metrics[s]
        print(f"{s}: P={p:.4f} R={r:.4f} F1={f1:.4f}")

    print(f"\nFinal test accuracy: {acc:.4f}")
    print(f"Final test macro precision / recall / f1: {macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}")

    try:
        save_confusion_matrix_plot(
            confusion,
            STATES,
            Path("./P1/output/confusion_matrix.png")
        )
    except Exception:
        print("混淆矩阵图保存失败，已跳过。")

    print("\n===== 预测样例 =====")
    for i, ex in enumerate(examples, 1):
        print(f"\n样例 {i}")
        print("原句:", ex["sentence"])
        print("真实分词:", "/".join(ex["true_words"]))
        print("预测分词:", "/".join(ex["pred_words"]))
        print("真实标签:", " ".join(ex["true_tags"]))
        print("预测标签:", " ".join(ex["pred_tags"]))


if __name__ == "__main__":
    main()
