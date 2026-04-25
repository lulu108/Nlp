from __future__ import annotations

from threading import Lock
from typing import Any


HANLP_NER_MODEL_NAME = "MSRA_NER_ELECTRA_SMALL_ZH"

_HANLP_MODEL: Any | None = None
_HANLP_LOAD_ATTEMPTED = False
_HANLP_LOAD_ERROR: Exception | None = None
_MODEL_LOCK = Lock()

_RULE_BASED_PATTERNS = [
    ("\u5317\u4eac\u5927\u5b66", "ORG"),
    ("\u5317\u4eac", "LOC"),
    ("\u5c0f\u660e", "PER"),
]

_LABEL_MAP = {
    "PER": "PER",
    "PERSON": "PER",
    "NR": "PER",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "NS": "LOC",
    "GPE": "LOC",
    "FAC": "LOC",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "NT": "ORG",
    "COMPANY": "ORG",
    "INSTITUTION": "ORG",
}


def recognize_entities(text: str) -> list[dict[str, object]]:
    """Recognize named entities with HanLP and fall back to deterministic rules."""
    if not text:
        print("DEBUG empty text")
        return []

    model = _get_hanlp_model()
    print("DEBUG input text:", repr(text))
    print("DEBUG model is None:", model is None)

    if model is None:
        print("DEBUG using fallback rules because model is unavailable")
        return _recognize_entities_with_rules(text)

    try:
        print("DEBUG using HanLP path")
        result = _predict_with_hanlp(model, text)
        print("DEBUG HanLP final entities:", result)
        return result
    except Exception as exc:
        print("DEBUG HanLP prediction failed:", repr(exc))
        print("DEBUG using fallback rules after exception")
        return _recognize_entities_with_rules(text)

def _get_hanlp_model() -> Any | None:
    global _HANLP_MODEL, _HANLP_LOAD_ATTEMPTED, _HANLP_LOAD_ERROR

    if _HANLP_LOAD_ATTEMPTED:
        return _HANLP_MODEL

    with _MODEL_LOCK:
        if _HANLP_LOAD_ATTEMPTED:
            return _HANLP_MODEL

        _HANLP_LOAD_ATTEMPTED = True

        try:
            import hanlp

            pretrained = getattr(hanlp, "pretrained", None)
            ner_models = getattr(pretrained, "ner", None)
            model_id = getattr(ner_models, HANLP_NER_MODEL_NAME, None)
            if model_id is None:
                raise RuntimeError(
                    f"HanLP pretrained NER model '{HANLP_NER_MODEL_NAME}' is not available."
                )

            _HANLP_MODEL = hanlp.load(model_id)
        except Exception as exc:
            _HANLP_LOAD_ERROR = exc
            _HANLP_MODEL = None

        return _HANLP_MODEL


def _predict_with_hanlp(model: Any, text: str) -> list[dict[str, object]]:
    raw_entities = model(list(text))
    print("DEBUG raw_entities:", raw_entities)

    entities: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for item in _iter_hanlp_entities(raw_entities):
        print("DEBUG raw item:", item)
        entity = _normalize_entity(item, text)
        print("DEBUG normalized entity:", entity)
        if entity is None:
            continue

        entity_key = (
            entity["text"],
            entity["label"],
            entity["start"],
            entity["end"],
        )
        if entity_key in seen:
            continue

        seen.add(entity_key)
        entities.append(entity)

    entities.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
    return entities


def _iter_hanlp_entities(raw_output: Any) -> list[Any]:
    if raw_output is None:
        return []

    if isinstance(raw_output, dict):
        if "ner" in raw_output:
            raw_output = raw_output["ner"]
        else:
            return []
    elif hasattr(raw_output, "get"):
        try:
            ner_output = raw_output.get("ner")
        except Exception:
            ner_output = None
        if ner_output is not None:
            raw_output = ner_output

    if isinstance(raw_output, list):
        return raw_output
    if isinstance(raw_output, tuple):
        return list(raw_output)
    return []


def _normalize_entity(item: Any, text: str) -> dict[str, object] | None:
    entity_text: str | None = None
    label: str | None = None
    start: int | None = None
    end: int | None = None

    if isinstance(item, dict):
        entity_text = item.get("text")
        label = item.get("label")
        start = item.get("start")
        end = item.get("end")
    elif isinstance(item, (list, tuple)):
        if len(item) == 4:
            entity_text, label, start, end = item
        elif len(item) == 3:
            label, start, end = item
        else:
            return None
    else:
        return None

    if not isinstance(label, str):
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end <= start:
        return None

    if not isinstance(entity_text, str) or not entity_text:
        if end > len(text):
            return None
        entity_text = text[start:end]

    if end > len(text) or text[start:end] != entity_text:
        span = _locate_entity_span(text, entity_text, start_hint=max(start, 0))
        if span is None:
            span = _locate_entity_span(text, entity_text, start_hint=0)
        if span is None:
            return None
        start, end = span

    return {
        "text": entity_text,
        "label": _normalize_label(label),
        "start": start,
        "end": end,
    }


def _normalize_label(label: str) -> str:
    normalized = label.strip().upper()

    for prefix in ("B-", "I-", "M-", "E-", "S-"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    return _LABEL_MAP.get(normalized, normalized)


def _locate_entity_span(
    text: str, entity_text: str, start_hint: int = 0
) -> tuple[int, int] | None:
    start = text.find(entity_text, start_hint)
    if start < 0:
        return None
    return start, start + len(entity_text)


def _recognize_entities_with_rules(text: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []

    for phrase, label in _RULE_BASED_PATTERNS:
        search_from = 0
        while True:
            start = text.find(phrase, search_from)
            if start < 0:
                break

            candidates.append(
                {
                    "text": phrase,
                    "label": label,
                    "start": start,
                    "end": start + len(phrase),
                }
            )
            search_from = start + len(phrase)

    candidates.sort(
        key=lambda item: (
            int(item["start"]),
            -len(str(item["text"])),
            str(item["label"]),
        )
    )

    entities: list[dict[str, object]] = []
    for candidate in candidates:
        overlaps = any(
            not (
                int(candidate["end"]) <= int(entity["start"])
                or int(candidate["start"]) >= int(entity["end"])
            )
            for entity in entities
        )
        if not overlaps:
            entities.append(candidate)

    entities.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
    return entities
