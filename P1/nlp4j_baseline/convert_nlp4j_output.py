# -*- coding: utf-8 -*-
"""Convert NLP4J TSV output into the unified P1 baseline format.

Default input:
    P1/nlp4j_baseline/sample_output.txt

Outputs:
    P1/nlp4j_baseline/output/nlp4j_result.tsv
    P1/nlp4j_baseline/output/nlp4j_entities.tsv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "sample_output.txt"
OUTPUT_DIR = BASE_DIR / "output"
RESULT_TSV = OUTPUT_DIR / "nlp4j_result.tsv"
ENTITIES_TSV = OUTPUT_DIR / "nlp4j_entities.tsv"


def read_text_auto(path: Path) -> str:
    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "gbk"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    raise ValueError(f"无法读取文件编码: {path}")


def parse_rows(text: str, source: Path) -> List[Tuple[int, str, str, str]]:
    rows: List[Tuple[int, str, str, str]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError(f"格式错误: {source} 第 {line_no} 行应为 4 列(sentence_id, token, pos, entity)，实际为 {len(parts)} 列")
        sentence_id_text, token, pos, entity = parts
        try:
            sentence_id = int(sentence_id_text)
        except Exception as exc:
            raise ValueError(f"格式错误: {source} 第 {line_no} 行的 sentence_id 不是整数: {sentence_id_text}") from exc
        if not token:
            raise ValueError(f"格式错误: {source} 第 {line_no} 行 token 为空")
        if not pos:
            raise ValueError(f"格式错误: {source} 第 {line_no} 行 pos 为空")
        if entity not in {"PER", "LOC", "ORG", "O"}:
            raise ValueError(f"格式错误: {source} 第 {line_no} 行 entity 必须是 PER/LOC/ORG/O，当前为 {entity}")
        rows.append((sentence_id, token, pos, entity))
    if not rows:
        raise ValueError(f"输入文件没有可解析的数据: {source}")
    return rows


def build_entities(rows: Sequence[Tuple[int, str, str, str]]) -> List[Tuple[int, str, str]]:
    entities: List[Tuple[int, str, str]] = []
    current_sentence = None
    current_type = None
    current_tokens: List[str] = []

    def flush() -> None:
        nonlocal current_sentence, current_type, current_tokens
        if current_sentence is not None and current_type in {"PER", "LOC", "ORG"} and current_tokens:
            entities.append((current_sentence, current_type, "".join(current_tokens)))
        current_sentence = None
        current_type = None
        current_tokens = []

    for sentence_id, token, _pos, entity in rows:
        if current_sentence is None:
            current_sentence = sentence_id
        if sentence_id != current_sentence:
            flush()
            current_sentence = sentence_id
        if entity == "O":
            flush()
            current_sentence = sentence_id
            continue
        if current_type == entity:
            current_tokens.append(token)
        else:
            flush()
            current_sentence = sentence_id
            current_type = entity
            current_tokens = [token]

    flush()
    return entities


def write_tsv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(item) for item in row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NLP4J TSV output into P1 baseline outputs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="输入 TSV 文件路径，默认读取 P1/nlp4j_baseline/sample_output.txt",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    text = read_text_auto(input_path)
    rows = parse_rows(text, input_path)
    entities = build_entities(rows)

    result_rows = [(sid, token, pos, entity) for sid, token, pos, entity in rows]
    entity_rows = [(sid, entity_type, entity_text) for sid, entity_type, entity_text in entities]

    write_tsv(RESULT_TSV, ["sentence_id", "token", "pos", "entity"], result_rows)
    write_tsv(ENTITIES_TSV, ["sentence_id", "entity_type", "entity_text"], entity_rows)

    print(f"Saved: {RESULT_TSV}")
    print(f"Saved: {ENTITIES_TSV}")


if __name__ == "__main__":
    main()
