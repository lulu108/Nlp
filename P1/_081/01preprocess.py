import re
from pathlib import Path
from typing import List

INPUT_PATH = Path("./P1/input/data.txt")
OUTPUT_PATH = Path("./P1/intermediate/news_clean.txt")

SENT_SPLIT_RE = re.compile(r"([^。！？；]*[。！？；])")
WHITESPACE_RE = re.compile(r"\s+")
INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# Simple noise heuristics for news QA boilerplate and template-like lines.
NOISE_PATTERNS = [
    re.compile(r"^有记者问"),
    re.compile(r"^记者问"),
    re.compile(r"^请问"),
    re.compile(r"^问："),
    re.compile(r"^答："),
]

# Common dateline or source prefixes to strip when they are purely formal.
PREFIX_PATTERNS = [
    re.compile(r"^央视网消息[:：]\s*"),
    re.compile(r"^新华社[^，。]*电\s*\(.*?\)"),
    re.compile(r"^新华社[^，。]*电\s*"),
    re.compile(r"^中新社[^，。]*电\s*"),
    re.compile(r"^当地时间\d{1,2}日[，,:：]\s*"),
]


def module1_read_and_clean(input_path: Path) -> str:
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r", "\n")
    text = INVISIBLE_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def module2_split_sentences(text: str) -> List[str]:
    parts = SENT_SPLIT_RE.findall(text)
    if not parts:
        parts = [text]
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


def _strip_prefix(sentence: str) -> str:
    cleaned = sentence
    for pat in PREFIX_PATTERNS:
        cleaned = pat.sub("", cleaned).strip()
    return cleaned


def _is_noise(sentence: str) -> bool:
    if all(ch.isdigit() for ch in sentence):
        return True
    if re.fullmatch(r"[()（）\[\]【】]+", sentence):
        return True
    for pat in NOISE_PATTERNS:
        if pat.search(sentence):
            return True
    return False


def module3_filter_sentences(sentences: List[str]) -> List[str]:
    filtered: List[str] = []
    for s in sentences:
        s = _strip_prefix(s)
        if not s:
            continue
        length = len(s)
        if length < 6:
            continue
        if length > 80:
            continue
        if _is_noise(s):
            continue
        filtered.append(s)
    return filtered


def module4_deduplicate(sentences: List[str]) -> List[str]:
    seen = set()
    unique = []
    for s in sentences:
        if s in seen:
            continue
        seen.add(s)
        unique.append(s)
    return unique


def module5_save(sentences: List[str], output_path: Path, raw_count: int) -> None:
    output_path.write_text("\n".join(sentences), encoding="utf-8")
    print(f"Before filter: {raw_count}")
    print(f"After filter:  {len(sentences)}")


def main() -> None:
    raw_text = module1_read_and_clean(INPUT_PATH)
    sentences = module2_split_sentences(raw_text)
    filtered = module3_filter_sentences(sentences)
    deduped = module4_deduplicate(filtered)
    module5_save(deduped, OUTPUT_PATH, raw_count=len(sentences))


if __name__ == "__main__":
    main()
