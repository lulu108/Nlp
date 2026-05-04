# -*- coding: utf-8 -*-
"""
Build rule-based dictionaries for the NLP4J baseline.

Inputs (optional but recommended):
- P1/nlp4j_baseline/dict/external/jieba_dict.txt
- P1/nlp4j_baseline/dict/external/thuocl/
- P1/output/ner_hanlp.txt
- P1/datasets/auto/train.txt, dev.txt, test.txt

Outputs:
- P1/nlp4j_baseline/dict/common_words.txt
- P1/nlp4j_baseline/dict/person.txt
- P1/nlp4j_baseline/dict/location.txt
- P1/nlp4j_baseline/dict/organization.txt
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
P1_DIR = BASE_DIR.parent
EXTERNAL_DIR = BASE_DIR / "dict" / "external"
THUOCL_DIR = EXTERNAL_DIR / "thuocl"

JIEBA_DICT = EXTERNAL_DIR / "jieba_dict.txt"
NER_HANLP = P1_DIR / "output" / "ner_hanlp.txt"
BMES_DIR = P1_DIR / "datasets" / "auto"

OUT_COMMON = BASE_DIR / "dict" / "common_words.txt"
OUT_PERSON = BASE_DIR / "dict" / "person.txt"
OUT_LOCATION = BASE_DIR / "dict" / "location.txt"
OUT_ORG = BASE_DIR / "dict" / "organization.txt"

MIN_COMMON_FREQ = 2


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ("utf-8-sig", "utf-8", "gbk", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def normalize_lines(lines: Iterable[str]) -> List[str]:
    return [line.strip() for line in lines if line and line.strip()]


def split_terms(text: str) -> List[str]:
    if not text:
        return []
    for sep in ["，", ",", "、", ";", "；", "|", "/", "\\"]:
        text = text.replace(sep, " ")
    return [tok.strip() for tok in text.split() if tok.strip()]


def load_jieba_common() -> Set[str]:
    if not JIEBA_DICT.exists():
        return set()
    words: Set[str] = set()
    for line in normalize_lines(JIEBA_DICT.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if line.startswith("#"):
            continue
        word = line.split()[0].strip()
        if word:
            words.add(word)
    return words


def find_thuocl_files() -> Tuple[List[Path], List[Path]]:
    if not THUOCL_DIR.exists():
        return [], []
    files = [p for p in THUOCL_DIR.iterdir() if p.is_file()]
    loc_files = [p for p in files if any(key in p.name.lower() for key in ["loc", "place", "location"]) or "地名" in p.name]
    person_files = [p for p in files if any(key in p.name.lower() for key in ["person", "name"]) or "人名" in p.name]
    return loc_files, person_files


def load_thuocl_terms(paths: List[Path]) -> Set[str]:
    words: Set[str] = set()
    for path in paths:
        text = read_text(path)
        for line in normalize_lines(text.splitlines()):
            word = line.split()[0].strip()
            if word:
                words.add(word)
    return words


def load_hanlp_entities() -> Tuple[Set[str], Set[str], Set[str]]:
    text = read_text(NER_HANLP)
    if not text:
        return set(), set(), set()

    person: Set[str] = set()
    location: Set[str] = set()
    organization: Set[str] = set()

    for block in [b.strip() for b in text.split("\n\n") if b.strip()]:
        for line in normalize_lines(block.splitlines()):
            if line.startswith("人名:"):
                person.update(split_terms(line.replace("人名:", "", 1).strip()))
            elif line.startswith("地名:"):
                location.update(split_terms(line.replace("地名:", "", 1).strip()))
            elif line.startswith("机构名:"):
                organization.update(split_terms(line.replace("机构名:", "", 1).strip()))
    return person, location, organization


def iter_bmes_files() -> List[Path]:
    candidates = [BMES_DIR / "train.txt", BMES_DIR / "dev.txt", BMES_DIR / "test.txt"]
    return [p for p in candidates if p.exists()]


def parse_bmes(path: Path) -> Tuple[Counter, Dict[str, Set[str]]]:
    text = read_text(path)
    if not text:
        return Counter(), {"PER": set(), "LOC": set(), "ORG": set()}

    counter: Counter = Counter()
    entities: Dict[str, Set[str]] = {"PER": set(), "LOC": set(), "ORG": set()}

    current_word: List[str] = []
    current_tag: Optional[str] = None

    def flush_word():
        nonlocal current_word, current_tag
        if not current_word:
            return
        word = "".join(current_word)
        counter[word] += 1
        if current_tag in entities:
            entities[current_tag].add(word)
        current_word = []
        current_tag = None

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            flush_word()
            continue
        parts = raw.split()
        if len(parts) < 2:
            continue
        token = parts[0].strip()
        tag = parts[-1].strip()

        if tag == "O":
            flush_word()
            counter[token] += 1
            continue

        if "-" in tag:
            prefix, label = tag.split("-", 1)
        else:
            prefix, label = tag, ""

        if prefix == "S":
            flush_word()
            counter[token] += 1
            if label in entities:
                entities[label].add(token)
        elif prefix == "B":
            flush_word()
            current_word = [token]
            current_tag = label
        elif prefix == "M":
            if current_word:
                current_word.append(token)
        elif prefix == "E":
            if current_word:
                current_word.append(token)
                flush_word()
        else:
            flush_word()
            counter[token] += 1

    flush_word()
    return counter, entities


def filter_common_words(counter: Counter, min_freq: int) -> Set[str]:
    return {word for word, count in counter.items() if count >= min_freq and len(word) >= 2}


def merge_words(*groups: Iterable[str]) -> Set[str]:
    merged: Set[str] = set()
    for group in groups:
        merged.update([x for x in group if x])
    return merged


def write_sorted(path: Path, words: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_words = sorted(set(words), key=lambda w: (-len(w), w))
    path.write_text("\n".join(sorted_words) + "\n", encoding="utf-8")


def main() -> None:
    common_words = set()
    person_words = set()
    location_words = set()
    organization_words = set()

    common_words.update(load_jieba_common())

    loc_files, person_files = find_thuocl_files()
    location_words.update(load_thuocl_terms(loc_files))
    person_words.update(load_thuocl_terms(person_files))

    hanlp_person, hanlp_location, hanlp_org = load_hanlp_entities()
    person_words.update(hanlp_person)
    location_words.update(hanlp_location)
    organization_words.update(hanlp_org)

    bmes_counter = Counter()
    bmes_entities: Dict[str, Set[str]] = {"PER": set(), "LOC": set(), "ORG": set()}
    for path in iter_bmes_files():
        counter, entities = parse_bmes(path)
        bmes_counter.update(counter)
        for label in bmes_entities:
            bmes_entities[label].update(entities.get(label, set()))

    common_words.update(filter_common_words(bmes_counter, MIN_COMMON_FREQ))
    person_words.update(bmes_entities["PER"])
    location_words.update(bmes_entities["LOC"])
    organization_words.update(bmes_entities["ORG"])

    existing_person = read_text(OUT_PERSON).splitlines() if OUT_PERSON.exists() else []
    existing_location = read_text(OUT_LOCATION).splitlines() if OUT_LOCATION.exists() else []
    existing_org = read_text(OUT_ORG).splitlines() if OUT_ORG.exists() else []

    person_words = merge_words(person_words, normalize_lines(existing_person))
    location_words = merge_words(location_words, normalize_lines(existing_location))
    organization_words = merge_words(organization_words, normalize_lines(existing_org))

    common_words = {w for w in common_words if len(w) >= 2}

    write_sorted(OUT_COMMON, common_words)
    write_sorted(OUT_PERSON, person_words)
    write_sorted(OUT_LOCATION, location_words)
    write_sorted(OUT_ORG, organization_words)

    print("Saved:", OUT_COMMON)
    print("Saved:", OUT_PERSON)
    print("Saved:", OUT_LOCATION)
    print("Saved:", OUT_ORG)


if __name__ == "__main__":
    main()
