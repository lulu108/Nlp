from __future__ import annotations

from algorithms.classifier import classify_text


def classify(text: str) -> tuple[str, float]:
	return classify_text(text)
