# -*- coding: utf-8 -*-
"""
P2/sample_thucnews_to_datatxt.py

功能：
1. 从 THUCNews 文件夹中抽样
2. 选取指定类别
3. 每类抽固定数量
4. 生成 P2/data/data.txt

输出格式：
    label\\ttext

运行方式：
    python P2\\sample_thucnews_to_datatxt.py
"""

import os
import re
import random
import unicodedata
from pathlib import Path


# =========================
# 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 这里改成你自己的 THUCNews 根目录
THUCNEWS_ROOT = Path(r"D:\Project\THUCNews")

OUTPUT_FILE = DATA_DIR / "data.txt"

# =========================
# 抽样配置
# =========================
TARGET_LABELS = ["体育", "财经", "科技", "娱乐"]
SAMPLE_PER_CLASS = 125
RANDOM_SEED = 42

# 最短文本长度（清洗后）
MIN_TEXT_LEN = 15


def normalize_text(text: str) -> str:
    """
    基础清洗：
    1. 全角转半角
    2. 去HTML
    3. 去URL
    4. 去多余空白
    5. 去掉大部分异常字符
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    # 去HTML标签
    text = re.sub(r"<[^>]+>", " ", text)

    # 去URL
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)

    # 去不可见字符
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # 换行/制表变空格
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # 去掉奇怪符号，尽量保留中文、英文、数字和常见标点
    text = re.sub(
        r"[^\u4e00-\u9fa5A-Za-z0-9，。！？；：、“”‘’（）()《》【】—…,.!?;:\-_/+%\s]",
        " ",
        text
    )

    # 压缩空白
    text = re.sub(r"\s+", " ", text).strip()

    return text


def read_text_file(file_path: Path) -> str:
    """
    尝试用 utf-8 读取，失败则回退 gbk
    """
    for enc in ["utf-8", "gbk", "utf-8-sig"]:
        try:
            with open(file_path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue

    # 最后兜底
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def collect_files_by_label(root_dir: Path, labels):
    """
    收集每个类别目录下的 txt 文件
    """
    label2files = {}

    for label in labels:
        label_dir = root_dir / label
        if not label_dir.exists() or not label_dir.is_dir():
            raise FileNotFoundError(f"未找到类别目录: {label_dir}")

        files = [p for p in label_dir.glob("*.txt") if p.is_file()]
        label2files[label] = files

    return label2files


def sample_texts(label2files, sample_per_class, min_text_len, seed=42):
    """
    每类抽样固定数量，并做简单过滤
    """
    random.seed(seed)
    samples = []

    for label, files in label2files.items():
        print(f"\n[{label}] 原始文件数: {len(files)}")

        # 先打乱，逐个读取过滤，直到凑够 sample_per_class
        random.shuffle(files)

        accepted = []
        rejected_empty = 0
        rejected_short = 0

        for fp in files:
            raw = read_text_file(fp)
            text = normalize_text(raw)

            if not text:
                rejected_empty += 1
                continue

            if len(text) < min_text_len:
                rejected_short += 1
                continue

            accepted.append((label, text, fp.name))

            if len(accepted) >= sample_per_class:
                break

        if len(accepted) < sample_per_class:
            raise ValueError(
                f"类别 {label} 有效样本不足。"
                f"需要 {sample_per_class} 条，实际只有 {len(accepted)} 条。"
            )

        print(f"[{label}] 空文本过滤: {rejected_empty}")
        print(f"[{label}] 过短文本过滤: {rejected_short}")
        print(f"[{label}] 最终抽样: {len(accepted)}")

        samples.extend(accepted)

    random.shuffle(samples)
    return samples


def write_data_txt(samples, output_file: Path):
    """
    写成 label\\ttext 格式
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for label, text, _fname in samples:
            # 为避免制表符破坏格式，这里替换掉文本中的 \t
            safe_text = text.replace("\t", " ")
            f.write(f"{label}\t{safe_text}\n")


def main():
    print("===== 从 THUCNews 抽样生成 data.txt 开始 =====")
    print(f"THUCNews 根目录: {THUCNEWS_ROOT}")
    print(f"目标类别: {TARGET_LABELS}")
    print(f"每类抽样数: {SAMPLE_PER_CLASS}")
    print(f"输出文件: {OUTPUT_FILE}")

    if not THUCNEWS_ROOT.exists():
        raise FileNotFoundError(f"THUCNews 根目录不存在: {THUCNEWS_ROOT}")

    label2files = collect_files_by_label(THUCNEWS_ROOT, TARGET_LABELS)

    samples = sample_texts(
        label2files=label2files,
        sample_per_class=SAMPLE_PER_CLASS,
        min_text_len=MIN_TEXT_LEN,
        seed=RANDOM_SEED,
    )

    write_data_txt(samples, OUTPUT_FILE)

    print("\n===== 抽样完成 =====")
    print(f"总样本数: {len(samples)}")
    for label in TARGET_LABELS:
        cnt = sum(1 for x in samples if x[0] == label)
        print(f"- {label}: {cnt}")

    print(f"\n已输出到: {OUTPUT_FILE}")
    print("格式: label\\ttext")

    print("\n前5条预览:")
    for i, (label, text, fname) in enumerate(samples[:5], start=1):
        preview = text[:60] + ("..." if len(text) > 60 else "")
        print(f"{i}. [{label}] {preview}  (source={fname})")

    print("\n===== 结束 =====")


if __name__ == "__main__":
    main()