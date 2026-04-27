#基础规范化处理
# 原始数据预处理”：

# 类别筛选
# 样本均衡抽样
# 文本基础清洗
# 最小长度过滤
# 标签标注
# 数据格式统一为 CSV

"""
从 THUCNews 中抽取 4 类文本（默认每类 150 条），输出到 P3/data。

默认类别：体育、科技、财经、教育
默认输出：P3/data/thucnews_4class_150.csv

设计目标：
1. 类别均衡抽样（每类固定数量）
2. 可复现（固定随机种子）
3. 文本质量过滤（最小长度阈值）
4. 默认采用“标题 + 正文前 N 字”策略

用法示例：
python P3/extract_thucnews_4class.py
python P3/extract_thucnews_4class.py --per-class 150 --seed 42
python P3/extract_thucnews_4class.py --text-mode body_only --body-chars 260
python P3/extract_thucnews_4class.py --min-clean-len 30
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple


# =========================
# 默认路径与参数
# =========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_THUC_ROOT = BASE_DIR.parent / "P2" / "THUCNews_fixed" / "THUCNews"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data"

DEFAULT_CATEGORIES = ["体育", "科技", "财经", "教育"]
DEFAULT_LABEL_MAP = {"体育": 0, "科技": 1, "财经": 2, "教育": 3}

DEFAULT_PER_CLASS = 150
DEFAULT_SEED = 42
DEFAULT_MIN_CLEAN_LEN = 30
DEFAULT_BODY_CHARS = 220


def normalize_text(text: str) -> str:
    """基础清洗，尽量保留中英文、数字与常见标点。"""
    if not text:
        return ""

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
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_text_file(path: Path) -> str:
    """兼容常见中文语料编码。"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_title_and_body(raw: str) -> Tuple[str, str]:
    """按行切分，优先把第一行视为标题。"""
    if not raw:
        return "", ""

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return "", ""

    if len(lines) == 1:
        return lines[0], ""

    title = lines[0]
    body = " ".join(lines[1:])
    return title, body


def build_text(raw: str, text_mode: str, body_chars: int) -> str:
    """
    组装文本：
    - title_body_head: 标题 + 正文前 body_chars 字（推荐）
    - body_only: 仅正文前 body_chars 字
    - full: 清洗后的全文
    """
    title_raw, body_raw = parse_title_and_body(raw)

    title = normalize_text(title_raw)
    body = normalize_text(body_raw)
    full = normalize_text(raw)

    if text_mode == "full":
        return full

    if text_mode == "body_only":
        if body:
            return body[:body_chars]
        return full[:body_chars]

    # 默认 title_body_head
    body_head = body[:body_chars] if body else ""
    if title and body_head:
        return f"{title} {body_head}".strip()
    if title:
        return title
    return full[:body_chars]


def sample_valid_rows_for_class(
    class_dir: Path,
    category: str,
    need_count: int,
    rng: random.Random,
    text_mode: str,
    body_chars: int,
    min_clean_len: int,
) -> Tuple[List[Tuple[str, str, str]], int]:
    """
    快速抽样：打乱文件后顺序读取，凑够 need_count 即停止。
    返回：(抽中的样本, 扫描过的文件数)
    """
    files = [p for p in class_dir.glob("*.txt") if p.is_file()]
    rng.shuffle(files)

    picked: List[Tuple[str, str, str]] = []
    scanned = 0

    for fp in files:
        scanned += 1
        raw = read_text_file(fp)
        text = build_text(raw=raw, text_mode=text_mode, body_chars=body_chars)
        if len(text) < min_clean_len:
            continue

        picked.append((category, text, fp.name))
        if len(picked) >= need_count:
            break

    return picked, scanned


def extract_balanced_dataset(
    thuc_root: Path,
    categories: List[str],
    label_map: Dict[str, int],
    per_class: int,
    seed: int,
    text_mode: str,
    body_chars: int,
    min_clean_len: int,
) -> List[Dict[str, str]]:
    """按类别均衡抽样并打标签。"""
    if not thuc_root.exists():
        raise FileNotFoundError(f"THUCNews 根目录不存在: {thuc_root}")

    rng = random.Random(seed)
    all_rows: List[Dict[str, str]] = []

    for cate in categories:
        class_dir = thuc_root / cate
        if not class_dir.exists() or not class_dir.is_dir():
            raise FileNotFoundError(f"未找到类别目录: {class_dir}")

        sampled_rows, scanned = sample_valid_rows_for_class(
            class_dir=class_dir,
            category=cate,
            need_count=per_class,
            rng=rng,
            text_mode=text_mode,
            body_chars=body_chars,
            min_clean_len=min_clean_len,
        )

        if len(sampled_rows) < per_class:
            raise ValueError(
                f"类别 {cate} 有效文本不足：需要 {per_class}，实际 {len(sampled_rows)}。"
            )

        for category, text, src in sampled_rows:
            all_rows.append(
                {
                    "label": str(label_map[category]),
                    "category": category,
                    "text": text,
                    "source_file": src,
                }
            )

        print(f"[{cate}] 已扫描文件数={scanned}，抽取={per_class}")

    rng.shuffle(all_rows)
    return all_rows


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "category", "text", "source_file"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 THUCNews 抽取四类均衡样本")
    parser.add_argument("--thuc-root", default=str(DEFAULT_THUC_ROOT), help="THUCNews 根目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="类别列表，逗号分隔，如：体育,科技,财经,教育",
    )
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS, help="每类抽样数量")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument(
        "--text-mode",
        choices=["title_body_head", "body_only", "full"],
        default="title_body_head",
        help="文本构建模式",
    )
    parser.add_argument(
        "--body-chars",
        type=int,
        default=DEFAULT_BODY_CHARS,
        help="正文截断长度（对 title_body_head/body_only 生效）",
    )
    parser.add_argument(
        "--min-clean-len",
        type=int,
        default=DEFAULT_MIN_CLEAN_LEN,
        help="清洗后最小长度，短于该值的样本会被过滤",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    thuc_root = Path(args.thuc_root)
    output_dir = Path(args.output_dir)
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    if set(categories) != set(DEFAULT_LABEL_MAP.keys()):
        raise ValueError(
            "当前脚本默认标签映射固定为 体育/科技/财经/教育。"
            "如需改类别，请同步修改 DEFAULT_LABEL_MAP。"
        )

    out_csv = output_dir / f"thucnews_4class_{args.per_class}.csv"

    print("===== THUCNews 四类抽取开始 =====")
    print(f"THUCNews 根目录: {thuc_root}")
    print(f"类别: {categories}")
    print(f"每类抽样: {args.per_class}")
    print(f"随机种子: {args.seed}")
    print(f"文本模式: {args.text_mode}")
    print(f"正文截断长度: {args.body_chars}")
    print(f"最小文本长度: {args.min_clean_len}")

    rows = extract_balanced_dataset(
        thuc_root=thuc_root,
        categories=categories,
        label_map=DEFAULT_LABEL_MAP,
        per_class=args.per_class,
        seed=args.seed,
        text_mode=args.text_mode,
        body_chars=args.body_chars,
        min_clean_len=args.min_clean_len,
    )

    write_csv(rows, out_csv)

    print("\n===== 抽取完成 =====")
    print(f"总样本数: {len(rows)}")
    for cate in categories:
        cnt = sum(1 for r in rows if r["category"] == cate)
        print(f"- {cate}: {cnt}")
    print(f"输出文件: {out_csv}")
    print("CSV 列: label, category, text, source_file")


if __name__ == "__main__":
    main()
