# -*- coding: utf-8 -*-
"""
Stage-B preprocessing for text clustering.

Input format:
    label<TAB>text

Main outputs (with optional tag suffix):
    raw_news_{tag}.csv
    news_stageB_{tag}.csv
    news_for_cluster_{tag}.csv
    stageB_top_tokens_{tag}.csv
    near_duplicates_removed_{tag}.csv
    process_log_{tag}.txt
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
import uuid
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_INPUT_FILE = DATA_DIR / "data440.txt"
MIN_TEXT_LEN = 15
NEAR_DUP_MIN_TOKENS = 12
NEAR_DUP_JACCARD = 0.85
NEAR_DUP_CONTAIN = 0.92

STOPWORD_FILES = [
    DATA_DIR / "stopwords.txt",
    DATA_DIR / "cn_stopwords.txt",
    DATA_DIR / "hit_stopwords.txt",
]

TEMPLATE_STOPWORD_FILES = [
    DATA_DIR / "template_stopwords.txt",
]

USER_DICT_FILES = [
    DATA_DIR / "user_dict.txt",
    DATA_DIR / "jieba_userdict.txt",
]

VALID_LABELS = {"体育", "科技", "教育", "房产"}

DEFAULT_STOPWORDS = {
    "的", "了", "和", "是", "在", "将", "都", "与", "及", "对", "被", "把", "等", "中", "上", "下",
    "但", "并", "而", "或", "还", "又", "很", "更", "最", "已", "已经", "进行", "表示", "相关", "记者",
    "目前", "今日", "今天", "昨日", "日前", "其中", "这个", "那个", "这些", "那些", "可以", "以及", "如果",
    "因为", "所以", "由于", "通过", "对于", "关于", "并且", "一个", "一种", "一些", "没有", "我们", "你们",
    "他们", "她们", "是否", "方面", "消息", "报道", "内容", "来源", "作者", "编辑", "发布", "更多", "点击",
    "查看", "表示", "成为", "表示", "指出", "认为", "显示",
}

DEFAULT_TEMPLATE_STOPWORDS = {
    "新浪", "体育讯", "娱乐讯", "科技讯", "房产", "教育频道", "我要评论", "进入论坛", "论坛", "博客",
    "微博", "微信", "二维码", "专题", "相册", "图片", "图集", "视频", "直播", "独家", "责编", "责任编辑",
    "来源", "编辑", "稿件", "声明", "地图搜索", "楼盘资料", "详细信息", "点击进入", "更多精彩", "相关专题",
}

KEEP_TOKENS = {"AI", "5G", "NBA", "CBA", "GDP", "APP", "WTA", "TD", "LTE", "3G", "4G"}

TEMPLATE_PATTERNS = [
    r"新浪[\u4e00-\u9fffA-Za-z]*讯",
    r"点击进入[^\s]{0,20}",
    r"进入[^\s]{0,20}专题",
    r"我要评论",
    r"详细信息",
    r"地图搜索",
    r"责任编辑[:：]?\S+",
    r"编辑[:：]?\S+",
    r"来源[:：]?\S+",
    r"本文来源[:：]?\S+",
    r"更多精彩[^\s]{0,20}",
]


def build_outputs(output_dir: Path, tag: str, cluster_output_override: str | None) -> dict[str, Path]:
    suffix = f"_{tag}" if tag else ""
    cluster_output = Path(cluster_output_override) if cluster_output_override else output_dir / f"news_for_cluster{suffix}.csv"
    return {
        "raw": output_dir / f"raw_news{suffix}.csv",
        "stageb": output_dir / f"news_stageB{suffix}.csv",
        "cluster": cluster_output,
        "log": output_dir / f"process_log{suffix}.txt",
        "top_tokens": output_dir / f"stageB_top_tokens{suffix}.csv",
        "near_dup": output_dir / f"near_duplicates_removed{suffix}.csv",
    }


def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(
        r"[^\u4e00-\u9fffA-Za-z0-9，。！？；：“”‘’（）《》【】、,.!?;:\-_/+%\s]",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def strip_template_text(text: str) -> str:
    cleaned = text
    for pattern in TEMPLATE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def load_lexicon(paths: list[Path], default_words: set[str]) -> tuple[set[str], list[str]]:
    words = set(default_words)
    loaded = []
    for fp in paths:
        if not fp.exists():
            continue
        loaded.append(str(fp))
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                word = line.split(maxsplit=1)[0]
                if word:
                    words.add(word)
    return words, loaded


def load_stopwords() -> tuple[set[str], list[str]]:
    return load_lexicon(STOPWORD_FILES, DEFAULT_STOPWORDS)


def load_template_stopwords() -> tuple[set[str], list[str]]:
    return load_lexicon(TEMPLATE_STOPWORD_FILES, DEFAULT_TEMPLATE_STOPWORDS)


def load_user_dicts() -> list[str]:
    loaded = []
    for fp in USER_DICT_FILES:
        if fp.exists():
            jieba.load_userdict(str(fp))
            loaded.append(str(fp))
    return loaded


def is_valid_token(tok: str) -> bool:
    tok = tok.strip()
    if not tok:
        return False
    if tok in KEEP_TOKENS:
        return True
    if re.fullmatch(r"[\W_]+", tok, flags=re.UNICODE):
        return False
    if re.fullmatch(r"\d+(\.\d+)?", tok):
        return False
    if len(tok) == 1:
        return False
    return True


def tokenize_and_filter(text: str, stopwords: set[str], template_words: set[str]) -> list[str]:
    tokens = jieba.lcut(text, cut_all=False)
    out = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok in KEEP_TOKENS:
            out.append(tok)
            continue
        if tok in stopwords or tok in template_words:
            continue
        if not is_valid_token(tok):
            continue
        out.append(tok)
    return out


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def token_ngram_set(tokens: list[str], n: int = 2) -> set[str]:
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def remove_near_duplicates(
    df: pd.DataFrame,
    *,
    token_col: str,
    scope: str,
    jaccard_threshold: float,
    contain_threshold: float,
    min_tokens: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scope == "off":
        return df.copy(), pd.DataFrame(
            columns=[
                "id",
                "label",
                "matched_id",
                "matched_label",
                "jaccard",
                "containment",
                "token_count",
            ]
        )

    keep_indices: list[int] = []
    kept_ngrams: list[set[str]] = []
    kept_labels: list[str] = []
    removed_rows: list[dict] = []

    for idx, row in df.iterrows():
        tokens = row[token_col]
        token_count = len(tokens)
        if token_count < min_tokens:
            keep_indices.append(idx)
            kept_ngrams.append(token_ngram_set(tokens))
            kept_labels.append(row["label"])
            continue

        current_ngrams = token_ngram_set(tokens)
        matched = None

        for pos, kept_idx in enumerate(keep_indices):
            if scope == "same_label" and row["label"] != kept_labels[pos]:
                continue

            prev_ngrams = kept_ngrams[pos]
            if not current_ngrams or not prev_ngrams:
                continue

            size_ratio = max(len(current_ngrams), len(prev_ngrams)) / max(1, min(len(current_ngrams), len(prev_ngrams)))
            if size_ratio > 1.6:
                continue

            inter = len(current_ngrams & prev_ngrams)
            if inter == 0:
                continue

            union = len(current_ngrams | prev_ngrams)
            jaccard = inter / max(1, union)
            containment = inter / max(1, min(len(current_ngrams), len(prev_ngrams)))

            if jaccard >= jaccard_threshold or containment >= contain_threshold:
                matched = (kept_idx, jaccard, containment)
                break

        if matched is None:
            keep_indices.append(idx)
            kept_ngrams.append(current_ngrams)
            kept_labels.append(row["label"])
            continue

        matched_idx, jaccard, containment = matched
        matched_row = df.loc[matched_idx]
        removed_rows.append(
            {
                "id": row["id"],
                "label": row["label"],
                "matched_id": matched_row["id"],
                "matched_label": matched_row["label"],
                "jaccard": round(float(jaccard), 4),
                "containment": round(float(containment), 4),
                "token_count": token_count,
            }
        )

    filtered = df.loc[keep_indices].copy().reset_index(drop=True)
    removed_df = pd.DataFrame(removed_rows)
    return filtered, removed_df


def parse_label_text_lines(raw_text: str, valid_labels: set[str]) -> tuple[list[dict], list[dict]]:
    records = []
    anomalies = []
    current = None

    for idx, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        matched = False
        for label in valid_labels:
            prefix = label + "\t"
            if line.startswith(prefix):
                text = line[len(prefix):].strip()
                current = {"line_no": idx, "label": label, "text": text}
                records.append(current)
                matched = True
                break

        if matched:
            continue

        if current is not None:
            current["text"] = (current["text"] + " " + line).strip()
        else:
            anomalies.append({"line_no": idx, "type": "head_orphan_line", "content": line[:120]})

    return records, anomalies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT_FILE), help="input txt file (label\\ttext)")
    parser.add_argument("--output-dir", default=str(DATA_DIR), help="output directory")
    parser.add_argument("--output", default="", help="override cluster output csv path")
    parser.add_argument("--tag", default="", help="output suffix tag, e.g. 550/750/1000")
    parser.add_argument("--min-text-len", type=int, default=MIN_TEXT_LEN, help="min normalized text length")
    parser.add_argument(
        "--near-dup-scope",
        choices=["off", "same_label", "all"],
        default="all",
        help="near-duplicate filtering scope",
    )
    parser.add_argument("--near-dup-jaccard", type=float, default=NEAR_DUP_JACCARD, help="near-duplicate jaccard threshold")
    parser.add_argument("--near-dup-contain", type=float, default=NEAR_DUP_CONTAIN, help="near-duplicate containment threshold")
    parser.add_argument("--near-dup-min-tokens", type=int, default=NEAR_DUP_MIN_TOKENS, help="min token count for near-duplicate detection")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outs = build_outputs(output_dir=output_dir, tag=args.tag, cluster_output_override=args.output or None)

    print("===== Stage-B preprocessing =====")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    raw = read_text_file(input_file)
    print(f"Read input: {input_file}")

    records, anomalies = parse_label_text_lines(raw, VALID_LABELS)
    print(f"Parsed records: {len(records)}")
    print(f"Head orphan lines: {len(anomalies)}")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No valid records parsed from input file.")

    df["text_raw"] = df["text"].astype(str)
    df["text"] = df["text_raw"].apply(normalize_text)
    df["text"] = df["text"].apply(strip_template_text).apply(normalize_text)
    df["text_compact"] = df["text"].apply(compact_text)

    before_empty = len(df)
    df = df[df["text"].str.strip() != ""].copy()
    after_empty = len(df)

    before_short = len(df)
    df = df[df["text"].str.len() >= args.min_text_len].copy()
    after_short = len(df)

    df["text_md5"] = df["text"].apply(md5_text)
    df["compact_md5"] = df["text_compact"].apply(md5_text)

    before_dup = len(df)
    df = df.drop_duplicates(subset=["text_md5"]).copy()
    df = df.drop_duplicates(subset=["compact_md5"]).copy()
    after_dup = len(df)

    df["id"] = [uuid.uuid4().hex for _ in range(len(df))]

    user_dicts = load_user_dicts()
    stopwords, loaded_sw = load_stopwords()
    template_words, loaded_tw = load_template_stopwords()

    print(f"Stopwords: {len(stopwords)}")
    print(f"Template words: {len(template_words)}")
    if loaded_sw:
        print("Loaded stopword files:")
        for item in loaded_sw:
            print(f"- {item}")
    if loaded_tw:
        print("Loaded template-word files:")
        for item in loaded_tw:
            print(f"- {item}")
    if user_dicts:
        print("Loaded jieba user dictionaries:")
        for item in user_dicts:
            print(f"- {item}")

    token_lists = []
    token_counts = []
    for text in df["text"]:
        toks = tokenize_and_filter(text, stopwords, template_words)
        token_lists.append(toks)
        token_counts.append(len(toks))

    df["tokens"] = token_lists
    df["token_count"] = token_counts
    df["text_cut"] = df["tokens"].apply(lambda x: " ".join(x))

    before_tok = len(df)
    df = df[df["token_count"] > 0].copy()
    after_tok = len(df)

    before_near_dup = len(df)
    df, near_dup_df = remove_near_duplicates(
        df,
        token_col="tokens",
        scope=args.near_dup_scope,
        jaccard_threshold=args.near_dup_jaccard,
        contain_threshold=args.near_dup_contain,
        min_tokens=args.near_dup_min_tokens,
    )
    after_near_dup = len(df)

    df_raw = df[["id", "label", "text"]].copy()
    df_raw.to_csv(outs["raw"], index=False, encoding="utf-8-sig")

    df_stageb = df[["id", "label", "text", "text_cut", "token_count"]].copy()
    df_stageb = df_stageb.rename(columns={"text": "text_norm"})
    df_stageb.to_csv(outs["stageb"], index=False, encoding="utf-8-sig")

    df_cluster = df[["id", "label", "text_cut"]].copy()
    df_cluster.to_csv(outs["cluster"], index=False, encoding="utf-8-sig")

    if near_dup_df.empty:
        near_dup_df = pd.DataFrame(
            columns=["id", "label", "matched_id", "matched_label", "jaccard", "containment", "token_count"]
        )
    near_dup_df.to_csv(outs["near_dup"], index=False, encoding="utf-8-sig")

    all_tokens = []
    for toks in df["tokens"]:
        all_tokens.extend(toks)
    top_df = pd.DataFrame(Counter(all_tokens).most_common(200), columns=["token", "freq"])
    top_df.to_csv(outs["top_tokens"], index=False, encoding="utf-8-sig")

    label_counts = df["label"].value_counts().to_dict()
    log_lines = [
        "===== Stage-B preprocessing log =====",
        f"input_file: {input_file}",
        f"parsed_records: {len(records)}",
        f"head_orphan_lines: {len(anomalies)}",
        f"removed_empty_text: {before_empty - after_empty}",
        f"removed_short_text(<{args.min_text_len}): {before_short - after_short}",
        f"removed_exact_duplicates: {before_dup - after_dup}",
        f"removed_zero_token_docs: {before_tok - after_tok}",
        f"near_dup_scope: {args.near_dup_scope}",
        f"removed_near_duplicates: {before_near_dup - after_near_dup}",
        f"near_dup_jaccard: {args.near_dup_jaccard}",
        f"near_dup_containment: {args.near_dup_contain}",
        f"near_dup_min_tokens: {args.near_dup_min_tokens}",
        f"stopword_count: {len(stopwords)}",
        f"template_word_count: {len(template_words)}",
        f"loaded_stopword_files: {loaded_sw}",
        f"loaded_template_files: {loaded_tw}",
        f"loaded_user_dicts: {user_dicts}",
        f"final_samples: {len(df)}",
        "label_distribution:",
    ]
    for key, value in label_counts.items():
        log_lines.append(f"- {key}: {value}")
    log_lines.append("outputs:")
    for key in ("raw", "stageb", "cluster", "top_tokens", "near_dup", "log"):
        log_lines.append(f"- {outs[key]}")
    outs["log"].write_text("\n".join(log_lines), encoding="utf-8")

    print("\n===== Cleaning summary =====")
    print(f"Removed empty text: {before_empty - after_empty}")
    print(f"Removed short text (<{args.min_text_len}): {before_short - after_short}")
    print(f"Removed exact duplicates: {before_dup - after_dup}")
    print(f"Removed zero-token docs: {before_tok - after_tok}")
    print(f"Removed near duplicates: {before_near_dup - after_near_dup}")
    print(f"Final samples: {len(df)}")
    print("\nLabel distribution:")
    for key, value in label_counts.items():
        print(f"- {key}: {value}")
    print("\nOutputs:")
    for key in ("raw", "stageb", "cluster", "top_tokens", "near_dup", "log"):
        print(f"- {outs[key]}")
    print("\n===== Done =====")


if __name__ == "__main__":
    main()
