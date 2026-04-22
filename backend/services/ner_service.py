from __future__ import annotations

from algorithms.ner import recognize_entities


def ner(text: str) -> list[dict[str, object]]:
	return recognize_entities(text)
