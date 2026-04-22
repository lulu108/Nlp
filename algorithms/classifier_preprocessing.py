from __future__ import annotations

from algorithms.tokenizer import tokenize_text


def normalize_raw_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def compose_classifier_source_text(*parts: object) -> str:
    normalized_parts: list[str] = []
    for part in parts:
        normalized = normalize_raw_text(part)
        if normalized:
            normalized_parts.append(normalized)
    return " ".join(normalized_parts)


def preprocess_classifier_text(text: object) -> str:
    normalized = normalize_raw_text(text)
    if not normalized:
        return ""

    tokens = [token.strip() for token in tokenize_text(normalized) if token.strip()]
    return " ".join(tokens)


def build_classifier_text(*parts: object) -> str:
    return preprocess_classifier_text(compose_classifier_source_text(*parts))
