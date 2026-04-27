# -*- coding: utf-8 -*-
"""
将原始 data.txt 处理成 BiLSTM-CRF / HMM 可用的 BMES 样本文件。

输入:
    data.txt

输出:
    data_bmes.txt
    格式:
    原句 \t 字序列(空格分隔) \t BMES标签序列(空格分隔) \t 分词结果(空格分隔)
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import re

from pyhanlp import HanLP


INPUT_PATH = Path("./P1/BiLSTMCRF/data/data.txt")
OUTPUT_PATH = Path("./P1/BiLSTMCRF/data/data_bmes.txt")

MIN_LEN = 6
MAX_LEN = 120
MAX_PUNCT_RATIO = 0.35

# 针对当前新闻域语料补充的长词、专名、多字实体。
USER_TERMS = [
    "中国人民银行",
    "证监会",
    "人民网",
    "海口海事局",
    "琼州海峡",
    "液化天然气",
    "个人所得税",
    "个人所得税法",
    "全国人大常委会",
    "生态环境部",
    "农业农村部",
    "长江经济带",
    "共同富裕",
    "中等收入",
    "居民收入",
    "党委政府",
    "各族群众",
    "连心桥",
    "城市更新",
    "基础设施",
    "超大规模",
    "充分发挥",
    "不白之冤",
    "不胜其烦",
    "除夕之夜",
    "安理会",
    "国际关系",
    "基本准则",
    "中新网",
    "燕新台",
    "黄愈丰",
    "中药材",
    "美好生活",
    "伊美谈判",
    "网络销售额",
    "渣打银行",
    "汇丰银行",
    "花旗银行",
    "证券公司",
    "自贸试验区",
    "发布公告",
]

# 只保留相对稳定的精确合并规则，避免过宽泛导致过拟合噪声。
MERGE_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (("中国", "人民银行"), "中国人民银行"),
    (("人民", "网"), "人民网"),
    (("证", "监会"), "证监会"),
    (("海口", "海事局"), "海口海事局"),
    (("琼州", "海峡"), "琼州海峡"),
    (("液化", "天然气"), "液化天然气"),
    (("个人", "所得税"), "个人所得税"),
    (("个人", "所得税", "法"), "个人所得税法"),
    (("全国人大", "常委会"), "全国人大常委会"),
    (("生态", "环境部"), "生态环境部"),
    (("农业", "农村部"), "农业农村部"),
    (("长江", "经济带"), "长江经济带"),
    (("共同", "富裕"), "共同富裕"),
    (("中等", "收入"), "中等收入"),
    (("居民", "收入"), "居民收入"),
    (("党委", "政府"), "党委政府"),
    (("各族", "群众"), "各族群众"),
    (("连心", "桥"), "连心桥"),
    (("城市", "更新"), "城市更新"),
    (("基础", "设施"), "基础设施"),
    (("超大", "规模"), "超大规模"),
    (("充分", "发挥"), "充分发挥"),
    (("不白", "之冤"), "不白之冤"),
    (("不胜", "其烦"), "不胜其烦"),
    (("除夕", "之夜"), "除夕之夜"),
    (("国际", "关系"), "国际关系"),
    (("基本", "准则"), "基本准则"),
    (("中", "新网"), "中新网"),
    (("中药", "材"), "中药材"),
    (("美好", "生活"), "美好生活"),
    (("伊美", "谈判"), "伊美谈判"),
    (("网络", "销售额"), "网络销售额"),
    (("发布", "公告"), "发布公告"),
    (("自", "贸试验区"), "自贸试验区"),
    (("证券", "公司"), "证券公司"),
    (("渣", "打银行"), "渣打银行"),
    (("汇丰", "银行"), "汇丰银行"),
    (("花旗", "银行"), "花旗银行"),
    (("黄愈", "丰"), "黄愈丰"),
]

NOISE_PATTERNS = [
    re.compile(r"^（[一二三四五六七八九十]+）"),
    re.compile(r"^第[一二三四五六七八九十0-9]+版"),
    re.compile(r"^责任编辑[:：]"),
    re.compile(r"^来源[:：]"),
]


def register_user_dict() -> None:
    """
    尝试把用户词典注册进 HanLP。
    """
    custom_dict = None
    try:
        custom_dict = HanLP.CustomDictionary
    except Exception:
        try:
            from pyhanlp import CustomDictionary  # type: ignore
            custom_dict = CustomDictionary
        except Exception:
            custom_dict = None

    if custom_dict is None:
        return

    for term in USER_TERMS:
        try:
            if hasattr(custom_dict, "add"):
                custom_dict.add(term)
            elif hasattr(custom_dict, "insert"):
                custom_dict.insert(term)
        except Exception:
            continue


def clean_text(text: str) -> str:
    """
    原始文本清洗。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ").replace("\t", " ")
    text = re.sub(r"[ ]+", " ", text)

    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if line == "图片":
            continue
        if "新华社发" in line:
            continue
        if any(pattern.match(line) for pattern in NOISE_PATTERNS):
            continue

        lines.append(line)

    return "\n".join(lines)


def split_to_sentences(text: str) -> List[str]:
    """
    按中文句末标点分句，保留句末标点。
    """
    sentences: List[str] = []
    parts = re.split(r"([。！？；])", text)

    buffer = ""
    for part in parts:
        if not part:
            continue
        if part in "。！？；":
            buffer += part
            sentence = buffer.strip()
            if sentence:
                sentences.append(sentence)
            buffer = ""
        else:
            buffer += part

    tail = buffer.strip()
    if tail:
        sentences.append(tail)

    return sentences


def normalize_sentence(sentence: str) -> str:
    """
    对单句做进一步规范化。
    """
    sentence = sentence.strip()
    sentence = sentence.replace(" ", "")
    sentence = sentence.replace("\u3000", "")
    sentence = sentence.replace("——", "—")
    sentence = re.sub(r"([，。！？；、])\1+", r"\1", sentence)
    sentence = sentence.strip("…")
    return sentence


def has_chinese(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def punct_ratio(text: str) -> float:
    punct_count = len(re.findall(r"[，。！？；、“”‘’（）()\[\]—\-…·：,:;!?\"']", text))
    return punct_count / max(len(text), 1)


def filter_sentence(sentence: str) -> bool:
    """
    过滤掉明显无效、过噪或几乎不含正文信息的句子。
    """
    length = len(sentence)
    if length < MIN_LEN or length > MAX_LEN:
        return False

    if not has_chinese(sentence):
        return False

    if punct_ratio(sentence) > MAX_PUNCT_RATIO:
        return False

    if re.match(r"^[，。！？；、“”‘’（）()\[\]—\-…]+$", sentence):
        return False

    return True


def word_to_bmes(word: str) -> List[str]:
    """
    一个词转为 BMES。
    """
    n = len(word)
    if n == 1:
        return ["S"]
    if n == 2:
        return ["B", "E"]
    return ["B"] + ["M"] * (n - 2) + ["E"]


def try_match_rule(words: Sequence[str], start: int) -> Optional[Tuple[int, str]]:
    for parts, merged in MERGE_RULES:
        end = start + len(parts)
        if tuple(words[start:end]) == parts:
            return end, merged
    return None


def merge_words(words: List[str]) -> List[str]:
    """
    对 HanLP 分词结果做精确规则合并。
    """
    merged: List[str] = []
    i = 0

    while i < len(words):
        matched = try_match_rule(words, i)
        if matched is not None:
            end, merged_word = matched
            merged.append(merged_word)
            i = end
            continue

        merged.append(words[i])
        i += 1

    return merged


def segment_sentence(sentence: str) -> List[str]:
    """
    HanLP 分词并应用后处理规则。
    """
    terms = HanLP.segment(sentence)
    words = [term.word for term in terms if str(term.word).strip()]
    return merge_words(words)


def sentence_to_sample(sentence: str):
    """
    句子 -> (chars, tags, words)
    """
    words = segment_sentence(sentence)

    chars: List[str] = []
    tags: List[str] = []
    for word in words:
        chars.extend(list(word))
        tags.extend(word_to_bmes(word))

    if len(chars) != len(tags):
        return None

    if len(chars) != len(sentence):
        return None

    return chars, tags, words


def main() -> None:
    register_user_dict()

    raw_text = INPUT_PATH.read_text(encoding="utf-8")
    cleaned_text = clean_text(raw_text)
    sentences = split_to_sentences(cleaned_text)

    final_sentences: List[str] = []
    seen = set()
    for sentence in sentences:
        sentence = normalize_sentence(sentence)
        if not filter_sentence(sentence):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        final_sentences.append(sentence)

    print(f"原始分句数: {len(sentences)}")
    print(f"过滤去重后句子数: {len(final_sentences)}")

    lines: List[str] = []
    valid_count = 0
    skipped_count = 0

    for sentence in final_sentences:
        sample = sentence_to_sample(sentence)
        if sample is None:
            skipped_count += 1
            continue

        chars, tags, words = sample
        lines.append(
            "\t".join(
                [
                    sentence,
                    " ".join(chars),
                    " ".join(tags),
                    " ".join(words),
                ]
            )
        )
        valid_count += 1

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"成功生成样本数: {valid_count}")
    print(f"跳过样本数: {skipped_count}")
    print(f"输出文件: {OUTPUT_PATH}")

    print("\n===== 前3条样例 =====")
    for line in lines[:3]:
        print(line)


if __name__ == "__main__":
    main()
