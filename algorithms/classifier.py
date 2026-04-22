from __future__ import annotations


def classify_text(text: str) -> tuple[str, float]:
    """Return deterministic baseline class and confidence."""
    value = text.strip()

    travel_kw = ["旅游", "景点", "北京", "观光"]
    tech_kw = ["科技", "人工智能", "大数据", "算法"]

    if any(k in value for k in travel_kw):
        return "旅游", 0.91
    if any(k in value for k in tech_kw):
        return "科技", 0.95

    return "其他", 0.8
