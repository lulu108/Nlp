from __future__ import annotations


def recognize_entities(text: str) -> list[dict[str, object]]:
    """Return lightweight deterministic entities based on simple phrase matching."""
    rules = [
        ("北京大学", "ORG"),
        ("北京", "LOC"),
        ("小明", "PER"),
    ]

    entities: list[dict[str, object]] = []
    for phrase, label in rules:
        start = text.find(phrase)
        if start >= 0:
            entities.append(
                {
                    "text": phrase,
                    "label": label,
                    "start": start,
                    "end": start + len(phrase),
                }
            )
    return entities
