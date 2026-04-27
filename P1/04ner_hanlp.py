# -*- coding: utf-8 -*-
"""
Use HanLP NER on test sentences and extract:
persons (nr), locations (ns), organizations (nt).
"""

from pathlib import Path
from typing import List, Dict

from pyhanlp import HanLP


TEST_PATH = Path("./P1/datasets/auto/test.txt")
OUTPUT_PATH = Path("./P1/output/ner_hanlp.txt")

# Set to True if you only want sentences that contain any entity
ONLY_WITH_ENTITY = False


def load_sentences(path: Path) -> List[str]:
    sentences: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if not parts:
            continue
        sentence = parts[0].strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def build_segmenter():
    segmenter = HanLP.newSegment()
    segmenter.enableNameRecognize(True)
    segmenter.enablePlaceRecognize(True)
    segmenter.enableOrganizationRecognize(True)
    return segmenter


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_entities(sentence: str, segmenter) -> Dict[str, List[str]]:
    persons: List[str] = []
    places: List[str] = []
    orgs: List[str] = []

    terms = segmenter.seg(sentence)
    for term in terms:
        word = str(term.word).strip()
        if not word:
            continue
        nature = str(term.nature) if term.nature is not None else ""
        if nature.startswith("nr"):
            persons.append(word)
        elif nature.startswith("ns"):
            places.append(word)
        elif nature.startswith("nt"):
            orgs.append(word)

    return {
        "person": unique_keep_order(persons),
        "location": unique_keep_order(places),
        "organization": unique_keep_order(orgs),
    }


def format_list(items: List[str]) -> str:
    return "、".join(items) if items else "无"


def main():
    sentences = load_sentences(TEST_PATH)
    segmenter = build_segmenter()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for i, sentence in enumerate(sentences, 1):
        entities = extract_entities(sentence, segmenter)
        if ONLY_WITH_ENTITY and not (
            entities["person"] or entities["location"] or entities["organization"]
        ):
            continue

        lines.append(f"句子{i}: {sentence}")
        lines.append(f"人名: {format_list(entities['person'])}")
        lines.append(f"地名: {format_list(entities['location'])}")
        lines.append(f"机构名: {format_list(entities['organization'])}")
        lines.append("")

    output_text = "\n".join(lines).strip() + "\n"
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")

    print(f"Sentences: {len(sentences)}")
    print(f"Saved NER result to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
