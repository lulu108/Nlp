#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
阶段2：训练前预处理 + 分层三划分 + TF-IDF特征构建

功能：
1. 读取阶段1清洗后的 CSV
2. 保留 label / category / text
3. 分层划分 train / dev / test
4. 对 train/dev/test 分别进行：
   - 中文分词
   - 去停用词
   - 生成 text_processed
5. 只用 train 拟合 TF-IDF
6. 分别转换 train/dev/test 特征
7. 保存处理结果与报告
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

import jieba
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "data" / "thucnews_4class_200_clean.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data"
DEFAULT_REPORT_TXT = BASE_DIR / "data" / "thucnews_4class_stage2_report.txt"
DEFAULT_STOPWORDS_PATH = BASE_DIR / "data" / "stopwords.txt"

REQUIRED_COLUMNS = ["label", "category", "text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段2：训练前预处理 + 三划分 + TF-IDF")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="阶段1清洗后CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_TXT), help="报告文件路径")
    parser.add_argument("--stopwords-path", default=str(DEFAULT_STOPWORDS_PATH), help="停用词表路径")

    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例，默认0.2")
    parser.add_argument(
        "--dev-size",
        type=float,
        default=0.2,
        help="开发集比例（相对于非测试集部分），默认0.2"
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")

    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF最大特征数")
    parser.add_argument("--min-df", type=int, default=2, help="词项最小文档频率")
    parser.add_argument("--max-df", type=float, default=0.95, help="词项最大文档频率比例")
    parser.add_argument("--ngram-max", type=int, default=1, help="ngram上界，1表示unigram，2表示1-2gram")

    parser.add_argument(
        "--keep-single-char",
        action="store_true",
        help="是否保留单字词，默认不保留",
    )
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}，当前列: {list(df.columns)}")


def load_stopwords(path: Path) -> Set[str]:
    if not path.exists():
        print(f"[提示] 停用词表不存在，将按空停用词表继续: {path}")
        return set()

    stopwords = set()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=enc) as f:
                for line in f:
                    w = line.strip()
                    if w:
                        stopwords.add(w)
            return stopwords
        except Exception:
            continue

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords.add(w)
    return stopwords


def is_valid_token(token: str, keep_single_char: bool = False) -> bool:
    token = token.strip()
    if not token:
        return False
    if not keep_single_char and len(token) == 1:
        return False
    return True


def preprocess_text(text: str, stopwords: Set[str], keep_single_char: bool = False) -> str:
    tokens = jieba.lcut(str(text).strip(), cut_all=False)
    processed_tokens: List[str] = []

    for tk in tokens:
        tk = tk.strip()
        if not is_valid_token(tk, keep_single_char=keep_single_char):
            continue
        if tk in stopwords:
            continue
        processed_tokens.append(tk)

    return " ".join(processed_tokens)


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    stopwords_path = Path(args.stopwords_path)

    if not input_csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("===== 阶段2：训练前预处理开始 =====")
    print(f"输入文件: {input_csv}")

    df = safe_read_csv(input_csv)
    ensure_required_columns(df)

    df = df[["label", "category", "text"]].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["category"] = df["category"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    print(f"样本总数: {len(df)}")
    print(f"类别分布: {df['category'].value_counts().to_dict()}")

    # 第一步：先切 test
    temp_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
        shuffle=True,
    )

    # 第二步：再从 temp 中切 train / dev
    train_df, dev_df = train_test_split(
        temp_df,
        test_size=args.dev_size,
        random_state=args.random_state,
        stratify=temp_df["label"],
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\n[1] 分层三划分完成")
    print(f"训练集样本数: {len(train_df)}")
    print(f"开发集样本数: {len(dev_df)}")
    print(f"测试集样本数: {len(test_df)}")
    print(f"训练集类别分布: {train_df['category'].value_counts().to_dict()}")
    print(f"开发集类别分布: {dev_df['category'].value_counts().to_dict()}")
    print(f"测试集类别分布: {test_df['category'].value_counts().to_dict()}")

    stopwords = load_stopwords(stopwords_path)
    print(f"\n[2] 停用词表加载完成: {len(stopwords)} 个词")

    print("\n[3] 开始处理训练集文本...")
    train_df["text_processed"] = train_df["text"].apply(
        lambda x: preprocess_text(x, stopwords=stopwords, keep_single_char=args.keep_single_char)
    )

    print("[4] 开始处理开发集文本...")
    dev_df["text_processed"] = dev_df["text"].apply(
        lambda x: preprocess_text(x, stopwords=stopwords, keep_single_char=args.keep_single_char)
    )

    print("[5] 开始处理测试集文本...")
    test_df["text_processed"] = test_df["text"].apply(
        lambda x: preprocess_text(x, stopwords=stopwords, keep_single_char=args.keep_single_char)
    )

    train_empty_processed = int(train_df["text_processed"].eq("").sum())
    dev_empty_processed = int(dev_df["text_processed"].eq("").sum())
    test_empty_processed = int(test_df["text_processed"].eq("").sum())

    print(f"训练集处理后空文本数: {train_empty_processed}")
    print(f"开发集处理后空文本数: {dev_empty_processed}")
    print(f"测试集处理后空文本数: {test_empty_processed}")

    if train_empty_processed > 0:
        train_df = train_df[train_df["text_processed"] != ""].reset_index(drop=True)
    if dev_empty_processed > 0:
        dev_df = dev_df[dev_df["text_processed"] != ""].reset_index(drop=True)
    if test_empty_processed > 0:
        test_df = test_df[test_df["text_processed"] != ""].reset_index(drop=True)

    ngram_range = (1, args.ngram_max)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b",
    )

    print("\n[6] 使用训练集拟合TF-IDF...")
    X_train = vectorizer.fit_transform(train_df["text_processed"])
    X_dev = vectorizer.transform(dev_df["text_processed"])
    X_test = vectorizer.transform(test_df["text_processed"])

    y_train = train_df["label"].to_numpy()
    y_dev = dev_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"词表大小: {len(vectorizer.vocabulary_)}")

    train_csv = output_dir / "train.csv"
    dev_csv = output_dir / "dev.csv"
    test_csv = output_dir / "test.csv"

    x_train_path = output_dir / "X_train_tfidf.npz"
    x_dev_path = output_dir / "X_dev_tfidf.npz"
    x_test_path = output_dir / "X_test_tfidf.npz"

    y_train_path = output_dir / "y_train.npy"
    y_dev_path = output_dir / "y_dev.npy"
    y_test_path = output_dir / "y_test.npy"

    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"

    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    dev_df.to_csv(dev_csv, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")

    save_npz(x_train_path, X_train)
    save_npz(x_dev_path, X_dev)
    save_npz(x_test_path, X_test)

    np.save(y_train_path, y_train)
    np.save(y_dev_path, y_dev)
    np.save(y_test_path, y_test)

    joblib.dump(vectorizer, vectorizer_path)

    report_lines = []
    report_lines.append("===== 阶段2：训练前预处理报告（三划分版） =====")
    report_lines.append(f"输入文件: {input_csv}")
    report_lines.append(f"总样本数: {len(df)}")
    report_lines.append("")

    report_lines.append("[1] 分层三划分")
    report_lines.append(f"训练集样本数: {len(train_df)}")
    report_lines.append(f"开发集样本数: {len(dev_df)}")
    report_lines.append(f"测试集样本数: {len(test_df)}")
    report_lines.append(f"训练集类别分布: {train_df['category'].value_counts().to_dict()}")
    report_lines.append(f"开发集类别分布: {dev_df['category'].value_counts().to_dict()}")
    report_lines.append(f"测试集类别分布: {test_df['category'].value_counts().to_dict()}")
    report_lines.append("")

    report_lines.append("[2] 分词与停用词过滤")
    report_lines.append(f"停用词数量: {len(stopwords)}")
    report_lines.append(f"训练集处理后空文本数: {train_empty_processed}")
    report_lines.append(f"开发集处理后空文本数: {dev_empty_processed}")
    report_lines.append(f"测试集处理后空文本数: {test_empty_processed}")
    report_lines.append("")

    report_lines.append("[3] TF-IDF特征")
    report_lines.append(f"max_features: {args.max_features}")
    report_lines.append(f"min_df: {args.min_df}")
    report_lines.append(f"max_df: {args.max_df}")
    report_lines.append(f"ngram_range: {ngram_range}")
    report_lines.append(f"X_train shape: {X_train.shape}")
    report_lines.append(f"X_dev shape: {X_dev.shape}")
    report_lines.append(f"X_test shape: {X_test.shape}")
    report_lines.append(f"词表大小: {len(vectorizer.vocabulary_)}")
    report_lines.append("")

    report_lines.append("[4] 输出文件")
    report_lines.append(f"train.csv: {train_csv}")
    report_lines.append(f"dev.csv: {dev_csv}")
    report_lines.append(f"test.csv: {test_csv}")
    report_lines.append(f"X_train_tfidf.npz: {x_train_path}")
    report_lines.append(f"X_dev_tfidf.npz: {x_dev_path}")
    report_lines.append(f"X_test_tfidf.npz: {x_test_path}")
    report_lines.append(f"y_train.npy: {y_train_path}")
    report_lines.append(f"y_dev.npy: {y_dev_path}")
    report_lines.append(f"y_test.npy: {y_test_path}")
    report_lines.append(f"tfidf_vectorizer.joblib: {vectorizer_path}")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n===== 阶段2完成（三划分版） =====")
    print(f"train.csv: {train_csv}")
    print(f"dev.csv: {dev_csv}")
    print(f"test.csv: {test_csv}")
    print(f"X_train_tfidf.npz: {x_train_path}")
    print(f"X_dev_tfidf.npz: {x_dev_path}")
    print(f"X_test_tfidf.npz: {x_test_path}")
    print(f"y_train.npy: {y_train_path}")
    print(f"y_dev.npy: {y_dev_path}")
    print(f"y_test.npy: {y_test_path}")
    print(f"tfidf_vectorizer.joblib: {vectorizer_path}")
    print(f"报告文件: {report_path}")


if __name__ == "__main__":
    main()