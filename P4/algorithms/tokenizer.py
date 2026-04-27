from __future__ import annotations

from typing import List


def tokenize_text(text: str) -> List[str]:
    """Return deterministic baseline tokenization output for demo and tests."""
    normalized = text.strip()
    if not normalized:
        return []

    try:
        import jieba

        return [tok for tok in jieba.lcut(normalized) if tok != ""]
    except Exception:
        # Fallback: keep deterministic behavior even when jieba is unavailable.
        return [ch for ch in normalized if ch.strip()]
