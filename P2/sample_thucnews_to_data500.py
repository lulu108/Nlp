#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nested sampling for THUCNews clustering experiments.

Default design:
- labels: 体育 / 科技 / 教育 / 房产
- groups:
  - A: 440 = 4 x 110
  - B: 600 = 4 x 150
  - C: 800 = 4 x 200

Outputs by default:
- P2/data/data440.txt
- P2/data/data600.txt
- P2/data/data800.txt
- P2/data/sampling_manifest_nested.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE_DIR / "THUCNews_fixed" / "THUCNews"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data"
DEFAULT_SEED = 42
DEFAULT_MIN_TEXT_LEN = 20
DEFAULT_LABELS = ["体育", "科技", "教育", "房产"]
DEFAULT_GROUPS = [("440", 110), ("600", 150), ("800", 200)]  # (tag, per_class_count)
DEFAULT_PREFIX = "data"


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_group_spec(value: str) -> list[tuple[str, int]]:
    """
    Example:
    "440:110,600:150,800:200" -> [("440",110),("600",150),("800",200)]
    """
    groups: list[tuple[str, int]] = []
    for seg in value.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if ":" not in seg:
            raise ValueError(f"Invalid group spec segment: {seg}")
        tag, cnt = seg.split(":", 1)
        tag = tag.strip()
        cnt = int(cnt.strip())
        if cnt <= 0:
            raise ValueError(f"Per-class count must be > 0, got {cnt}")
        groups.append((tag, cnt))
    if not groups:
        raise ValueError("No valid groups were parsed.")
    return groups


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
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def sample_master_pool(
    root: Path,
    labels: list[str],
    max_per_class: int,
    seed: int,
    min_text_len: int,
) -> dict[str, list[tuple[str, str, str]]]:
    """
    Returns:
    {
      label: [(label, normalized_text, source_filename), ...]
    }
    """
    if not root.exists():
        raise FileNotFoundError(f"THUCNews root not found: {root}")

    rng = random.Random(seed)
    pool: dict[str, list[tuple[str, str, str]]] = {}

    for label in labels:
        class_dir = root / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        files = [p for p in class_dir.glob("*.txt") if p.is_file()]
        rng.shuffle(files)

        picked: list[tuple[str, str, str]] = []
        for fp in files:
            text = normalize_text(read_text(fp))
            if len(text) < min_text_len:
                continue
            picked.append((label, text, fp.name))
            if len(picked) >= max_per_class:
                break

        if len(picked) < max_per_class:
            raise ValueError(
                f"Class {label} has only {len(picked)} valid samples, "
                f"expected at least {max_per_class}."
            )
        pool[label] = picked

    return pool


def write_group_file(
    rows: list[tuple[str, str, str]],
    output_path: Path,
    seed: int,
    tag: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep nested set property while randomizing display order per group.
    rng = random.Random(seed + sum(ord(c) for c in tag))
    idx = list(range(len(rows)))
    rng.shuffle(idx)

    with output_path.open("w", encoding="utf-8") as f:
        for i in idx:
            label, text, _src = rows[i]
            f.write(f"{label}\t{text.replace(chr(9), ' ')}\n")


def write_manifest(
    manifest_path: Path,
    groups: list[tuple[str, int]],
    labels: list[str],
    outputs: dict[str, Path],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group_tag", "per_class", "total", "label", "output_file"],
        )
        writer.writeheader()
        for tag, n in groups:
            total = n * len(labels)
            for label in labels:
                writer.writerow(
                    {
                        "group_tag": tag,
                        "per_class": n,
                        "total": total,
                        "label": label,
                        "output_file": str(outputs[tag]),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="THUCNews root dir")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output directory")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="output filename prefix")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--min-text-len", type=int, default=DEFAULT_MIN_TEXT_LEN, help="minimum text length")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="comma-separated labels, e.g. 体育,科技,教育,房产",
    )
    parser.add_argument(
        "--groups",
        default="440:110,600:150,800:200",
        help="comma-separated group spec tag:per_class, e.g. 440:110,600:150,800:200",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    labels = parse_csv_list(args.labels)
    groups = parse_group_spec(args.groups)
    max_per_class = max(n for _tag, n in groups)

    pool = sample_master_pool(
        root=root,
        labels=labels,
        max_per_class=max_per_class,
        seed=args.seed,
        min_text_len=args.min_text_len,
    )

    outputs: dict[str, Path] = {}
    for tag, n in groups:
        rows: list[tuple[str, str, str]] = []
        for label in labels:
            rows.extend(pool[label][:n])  # nested sampling: A ⊂ B ⊂ C

        out_file = output_dir / f"{args.prefix}{tag}.txt"
        write_group_file(rows=rows, output_path=out_file, seed=args.seed, tag=tag)
        outputs[tag] = out_file

    manifest = output_dir / "sampling_manifest_nested.csv"
    write_manifest(manifest_path=manifest, groups=groups, labels=labels, outputs=outputs)

    print("Nested sampling completed.")
    print(f"labels: {labels}")
    print(f"groups: {groups}")
    for tag, n in groups:
        print(f"- {tag}: per_class={n}, total={n * len(labels)}, file={outputs[tag]}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
