# -*- coding: utf-8 -*-
"""
Prepare BiLSTM-CRF data directory.

Default:
- Source: P1/BiLSTMCRF/data/data.txt
- Target: P1/BiLSTMCRF/data/train.txt, dev.txt, test.txt
"""

from pathlib import Path
import random

DST_DIR = Path("./P1/BiLSTMCRF/data")

SOURCE_NAME = "data_bmes.txt"
TRAIN_NAME = "train.txt"
DEV_NAME = "dev.txt"
TEST_NAME = "test.txt"

TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42


def read_lines(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_lines(lines, train_ratio, dev_ratio, test_ratio):
    total = len(lines)
    if total == 0:
        raise ValueError("Empty dataset.")
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    train_size = int(total * train_ratio)
    dev_size = int(total * dev_ratio)
    test_size = total - train_size - dev_size

    if total >= 3:
        if dev_size == 0:
            dev_size = 1
            train_size = max(1, train_size - 1)
        if test_size == 0:
            test_size = 1
            train_size = max(1, train_size - 1)

    train_size = total - dev_size - test_size
    train_lines = lines[:train_size]
    dev_lines = lines[train_size:train_size + dev_size]
    test_lines = lines[train_size + dev_size:]
    return train_lines, dev_lines, test_lines


def main():
    src = DST_DIR / SOURCE_NAME
    lines = read_lines(src)

    random.seed(RANDOM_SEED)
    random.shuffle(lines)

    train_lines, dev_lines, test_lines = split_lines(
        lines, TRAIN_RATIO, DEV_RATIO, TEST_RATIO
    )

    DST_DIR.mkdir(parents=True, exist_ok=True)
    (DST_DIR / TRAIN_NAME).write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (DST_DIR / DEV_NAME).write_text("\n".join(dev_lines) + "\n", encoding="utf-8")
    (DST_DIR / TEST_NAME).write_text("\n".join(test_lines) + "\n", encoding="utf-8")

    print(f"Train: {len(train_lines)}")
    print(f"Dev:   {len(dev_lines)}")
    print(f"Test:  {len(test_lines)}")
    print(f"Saved to: {DST_DIR}")


if __name__ == "__main__":
    main()
