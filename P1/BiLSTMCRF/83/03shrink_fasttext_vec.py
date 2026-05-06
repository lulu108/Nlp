# -*- coding: utf-8 -*-
"""
从大规模 fastText 向量中裁剪出当前数据集需要的字符向量子集。

默认输入:
    P1/BiLSTMCRF/data/cc.zh.300.vec/cc.zh.300.vec
默认输出:
    P1/BiLSTMCRF/data/cc.zh.300.char_subset.vec

用法:
    python P1/BiLSTMCRF/03shrink_fasttext_vec.py
    python P1/BiLSTMCRF/03shrink_fasttext_vec.py --source <大向量路径> --out <输出路径>
"""

from pathlib import Path
import argparse

DATA_DIR = Path("./P1/BiLSTMCRF/data")
DEFAULT_SOURCE = DATA_DIR / "cc.zh.300.vec" / "cc.zh.300.vec"
DEFAULT_OUT = DATA_DIR / "cc.zh.300.char_subset.vec"
SPLIT_FILES = [DATA_DIR / "train.txt", DATA_DIR / "dev.txt", DATA_DIR / "test.txt"]


def collect_chars(paths):
    chars = set()
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chars.update(parts[1].split())
    return chars


def detect_header(first_parts):
    return len(first_parts) == 2 and first_parts[0].isdigit() and first_parts[1].isdigit()


def shrink_vec(source: Path, out_path: Path, required_chars):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    found = {}
    dim = None
    total_tokens = None

    with source.open("r", encoding="utf-8") as fin:
        first_parts = fin.readline().strip().split()
        has_header = detect_header(first_parts)

        if has_header:
            total_tokens = int(first_parts[0])
            dim = int(first_parts[1])
        else:
            if len(first_parts) >= 3:
                dim = len(first_parts) - 1
                token = first_parts[0]
                if token in required_chars:
                    found[token] = " ".join(first_parts[1:])

        for line in fin:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 3:
                continue
            token = parts[0]
            if token in required_chars and token not in found:
                found[token] = " ".join(parts[1:])
            if len(found) == len(required_chars):
                break

    if dim is None:
        raise ValueError("无法从向量文件推断维度，请检查文件格式。")

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write(f"{len(found)} {dim}\n")
        for ch in sorted(found.keys()):
            fout.write(f"{ch} {found[ch]}\n")

    hit = len(found)
    need = len(required_chars)
    miss_chars = sorted(required_chars - set(found.keys()))
    print(f"source: {source}")
    if total_tokens is not None:
        print(f"source_header: tokens={total_tokens}, dim={dim}")
    print(f"required_chars: {need}")
    print(f"matched_chars: {hit}")
    print(f"coverage: {hit / max(1, need) * 100:.2f}%")
    print(f"saved_to: {out_path}")
    if miss_chars:
        print("missing_sample:", "".join(miss_chars[:80]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE), help="fastText .vec 路径")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="输出子集向量路径")
    args = parser.parse_args()

    source = Path(args.source)
    out_path = Path(args.out)

    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    required_chars = collect_chars(SPLIT_FILES)
    if not required_chars:
        raise ValueError("未从 train/dev/test 收集到字符，请先准备数据集。")

    shrink_vec(source, out_path, required_chars)


if __name__ == "__main__":
    main()
