"""
功能：
1. 读取 news_clean.txt
2. 用 HanLP 分词
3. 将分词结果转成 BMES 标签
4. 划分训练集和测试集
5. 保存为 train.txt / test.txt

文件格式：
每行一个样本，格式为：
原句 \t 空格分隔的字序列 \t 空格分隔的BMES标签序列 \t 空格分隔的分词结果
"""

from pathlib import Path
from typing import List, Tuple
import random

# ====== 先安装 pyhanlp ======
# pip install pyhanlp
from pyhanlp import HanLP


INPUT_PATH = Path("./P1/test2/intermediate/news_clean.txt")
TRAIN_PATH = Path("./P1/test2/datasets/auto/train.txt")
TEST_PATH = Path("./P1/test2/datasets/auto/test.txt")

TEST_RATIO = 0.2
RANDOM_SEED = 42


def read_sentences(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sentences = [line.strip() for line in lines if line.strip()]
    return sentences


def segment_sentence(sentence: str) -> List[str]:
    """
    使用 HanLP 对句子分词
    返回词列表
    """
    sentence = clean_sentence_for_seg(sentence)
    terms = HanLP.segment(sentence)
    words = [term.word for term in terms if str(term.word).strip()]
    words = merge_words(words)
    return words


def clean_sentence_for_seg(sentence: str) -> str:
    """
    分词前清洗：
    1. 去掉首尾空格
    2. 去掉句中普通空格和全角空格
    """
    sentence = sentence.strip()
    sentence = sentence.replace(" ", "")
    sentence = sentence.replace("\u3000", "")
    return sentence


def merge_words(words: List[str]) -> List[str]:
    """
    对 HanLP 分词结果做少量规则合并
    """
    merged = []
    i = 0

    while i < len(words):
        # 五小 + 时 -> 五小时
        if i + 1 < len(words) and words[i] == "五小" and words[i + 1] == "时":
            merged.append("五小时")
            i += 2
            continue

        # 飞行 + 圈 -> 飞行圈
        if i + 1 < len(words) and words[i] == "飞行" and words[i + 1] == "圈":
            merged.append("飞行圈")
            i += 2
            continue

        # 高 + 收入 -> 高收入
        if i + 1 < len(words) and words[i] == "高" and words[i + 1] == "收入":
            merged.append("高收入")
            i += 2
            continue

        merged.append(words[i])
        i += 1

    return merged


def register_user_dict(words: List[str]) -> None:
    """
    尝试把用户词典词加入 HanLP 自定义词典（如可用则注册），失败则静默跳过。
    """
    try:
        from pyhanlp import CustomDictionary
    except Exception:
        return

    for w in words:
        if not w or not str(w).strip():
            continue
        try:
            # 兼容不同版本的 API：先尝试 add，再尝试 insert
            if hasattr(CustomDictionary, 'add'):
                CustomDictionary.add(w)
            elif hasattr(CustomDictionary, 'insert'):
                CustomDictionary.insert(w)
        except Exception:
            # 忽略注册失败
            continue


def word_to_bmes(word: str) -> List[str]:
    """
    将一个词转换成 BMES 标签
    """
    n = len(word)
    if n == 1:
        return ["S"]
    elif n == 2:
        return ["B", "E"]
    else:
        return ["B"] + ["M"] * (n - 2) + ["E"]


def sentence_to_char_and_tags(words: List[str]) -> Tuple[List[str], List[str]]:
    """
    把分词结果转成：
    1. 字序列
    2. BMES 标签序列
    """
    chars = []
    tags = []

    for word in words:
        chars.extend(list(word))
        tags.extend(word_to_bmes(word))

    assert len(chars) == len(tags), "字序列与标签序列长度不一致"
    return chars, tags


def build_samples(sentences: List[str]) -> List[Tuple[str, List[str], List[str], List[str]]]:
    """
    返回：
    [
        (原句, chars, tags, words),
        ...
    ]
    """
    samples = []
    for sentence in sentences:
        words = segment_sentence(sentence)
        chars, tags = sentence_to_char_and_tags(words)

        # 长度校验
        if len(sentence.replace(" ", "")) != len(chars):
            # 极少数情况下分词或原句中存在特殊空白字符，这里做个保护
            # 不直接报错，跳过异常样本
            continue

        samples.append((sentence, chars, tags, words))
    return samples


def train_test_split(samples: List[Tuple], test_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    shuffled = samples[:]
    random.shuffle(shuffled)

    test_size = int(len(shuffled) * test_ratio)
    test_samples = shuffled[:test_size]
    train_samples = shuffled[test_size:]
    return train_samples, test_samples


def save_samples(samples: List[Tuple[str, List[str], List[str], List[str]]], path: Path) -> None:
    """
    保存格式：
    原句 \t 字1 字2 ... \t 标签1 标签2 ... \t 词1 词2 ...
    """
    lines = []
    for sentence, chars, tags, words in samples:
        line = "\t".join([
            sentence,
            " ".join(chars),
            " ".join(tags),
            " ".join(words)
        ])
        lines.append(line)

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    sentences = read_sentences(INPUT_PATH)
    print(f"读取句子数: {len(sentences)}")

    samples = build_samples(sentences)
    print(f"成功生成样本数: {len(samples)}")

    train_samples, test_samples = train_test_split(
        samples, test_ratio=TEST_RATIO, seed=RANDOM_SEED
    )

    print(f"训练集样本数: {len(train_samples)}")
    print(f"测试集样本数: {len(test_samples)}")

    save_samples(train_samples, TRAIN_PATH)
    save_samples(test_samples, TEST_PATH)

    # 打印前3条看看
    print("\n===== 样例输出 =====")
    for i, (sentence, chars, tags, words) in enumerate(train_samples[:3]):
        print(f"\n样本 {i+1}")
        print("原句:", sentence)
        print("分词:", "/".join(words))
        print("字序列:", " ".join(chars))
        print("标签序列:", " ".join(tags))


if __name__ == "__main__":
    main()