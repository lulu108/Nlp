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
import math

TRAIN_PATH = Path("./P1/datasets/auto/train.txt")
TEST_PATH = Path("./P1/datasets/auto/test.txt")

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

    show_results = []

    for sample in samples:
        chars = sample["chars"]
        true_tags = sample["tags"]

        pred_tags = model.viterbi(chars)

        for t1, t2 in zip(true_tags, pred_tags):
            total_tags += 1
            if t1 == t2:
                correct_tags += 1

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
    return acc, show_results


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

    acc, examples = evaluate(model, test_samples)
    print(f"\n测试集标签级准确率: {acc:.4f}")

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