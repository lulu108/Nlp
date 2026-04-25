from __future__ import annotations

from threading import Lock
from typing import Any


HANLP_TOKENIZER_MODEL_NAME = "COARSE_ELECTRA_SMALL_ZH"
HANLP_NER_MODEL_NAME = "MSRA_NER_ELECTRA_SMALL_ZH"

_HANLP_TOKENIZER: Any | None = None
_HANLP_NER_MODEL: Any | None = None
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
        return []

    tokenizer, ner_model = _get_hanlp_models()

    if tokenizer is None or ner_model is None:
        return _recognize_entities_with_rules(text)

    try:
        return _predict_with_hanlp(tokenizer, ner_model, text)
    except Exception:
        return _recognize_entities_with_rules(text)


def _get_hanlp_models() -> tuple[Any | None, Any | None]:
    global _HANLP_TOKENIZER, _HANLP_NER_MODEL, _HANLP_LOAD_ATTEMPTED, _HANLP_LOAD_ERROR

    if _HANLP_LOAD_ATTEMPTED:
        return _HANLP_TOKENIZER, _HANLP_NER_MODEL

    with _MODEL_LOCK:
        if _HANLP_LOAD_ATTEMPTED:
            return _HANLP_TOKENIZER, _HANLP_NER_MODEL

        _HANLP_LOAD_ATTEMPTED = True

        try:
            import hanlp

            pretrained = getattr(hanlp, "pretrained", None)

            tok_models = getattr(pretrained, "tok", None)
            tok_model_id = getattr(tok_models, HANLP_TOKENIZER_MODEL_NAME, None)
            if tok_model_id is None:
                raise RuntimeError(
                    f"HanLP pretrained tokenizer model '{HANLP_TOKENIZER_MODEL_NAME}' is not available."
                )

            ner_models = getattr(pretrained, "ner", None)
            ner_model_id = getattr(ner_models, HANLP_NER_MODEL_NAME, None)
            if ner_model_id is None:
                raise RuntimeError(
                    f"HanLP pretrained NER model '{HANLP_NER_MODEL_NAME}' is not available."
                )

            _HANLP_TOKENIZER = hanlp.load(tok_model_id)
            _HANLP_NER_MODEL = hanlp.load(ner_model_id)
        except Exception as exc:
            _HANLP_LOAD_ERROR = exc
            _HANLP_TOKENIZER = None
            _HANLP_NER_MODEL = None

        return _HANLP_TOKENIZER, _HANLP_NER_MODEL


def _predict_with_hanlp(tokenizer: Any, ner_model: Any, text: str) -> list[dict[str, object]]:
    tokens = tokenizer(text)
    if not isinstance(tokens, list):
        return []

    token_offsets = _build_token_offsets(text, tokens)
    raw_entities = ner_model(tokens)

    entities: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for item in _iter_hanlp_entities(raw_entities):
        entity = _normalize_entity(item, text, tokens, token_offsets)
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


def _build_token_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0

    for token in tokens:
        if not isinstance(token, str) or not token:
            offsets.append((-1, -1))
            continue

        start = text.find(token, cursor)
        if start < 0:
            start = text.find(token)
        if start < 0:
            offsets.append((-1, -1))
            continue

        end = start + len(token)
        offsets.append((start, end))
        cursor = end

    return offsets


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


def _normalize_entity(
    item: Any,
    text: str,
    tokens: list[str],
    token_offsets: list[tuple[int, int]],
) -> dict[str, object] | None:
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

    token_start = start
    token_end = end
    is_token_span = _looks_like_token_span(
        token_start,
        token_end,
        token_offsets,
        tokens=tokens,
        entity_text=entity_text,
    )

    if is_token_span:
        if not isinstance(entity_text, str) or not entity_text:
            recovered = _entity_text_from_tokens(tokens, token_start, token_end)
            if recovered:
                entity_text = recovered

        span = _token_span_to_char_span(token_start, token_end, token_offsets)
        if span is None:
            return None
        start, end = span
        if not isinstance(entity_text, str) or not entity_text:
            entity_text = text[start:end]

    elif not isinstance(entity_text, str) or not entity_text:
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


def _looks_like_token_span(
    start: int,
    end: int,
    token_offsets: list[tuple[int, int]],
    tokens: list[str] | None = None,
    entity_text: str | None = None,
) -> bool:
    if not token_offsets:
        return False

    if start < 0 or end <= start:
        return False

    token_count = len(token_offsets)
    if not (start < token_count and end <= token_count):
        return False

    span_offsets = token_offsets[start:end]
    if not span_offsets or any(offset[0] < 0 or offset[1] < 0 for offset in span_offsets):
        return False

    if isinstance(entity_text, str) and entity_text:
        token_text = _entity_text_from_tokens(tokens or [], start, end)
        if token_text != entity_text:
            return False

    return True


def _token_span_to_char_span(
    token_start: int,
    token_end: int,
    token_offsets: list[tuple[int, int]],
) -> tuple[int, int] | None:
    if token_start < 0 or token_end <= token_start:
        return None
    if token_end > len(token_offsets):
        return None

    span_offsets = token_offsets[token_start:token_end]
    valid_offsets = [offset for offset in span_offsets if offset[0] >= 0 and offset[1] >= 0]
    if not valid_offsets:
        return None

    char_start = valid_offsets[0][0]
    char_end = valid_offsets[-1][1]
    return char_start, char_end


def _entity_text_from_tokens(tokens: list[str], token_start: int, token_end: int) -> str | None:
    if token_start < 0 or token_end <= token_start:
        return None
    if token_end > len(tokens):
        return None
    return "".join(tokens[token_start:token_end])


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
