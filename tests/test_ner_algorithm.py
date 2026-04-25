from algorithms import ner as ner_module


def test_recognize_entities_uses_hanlp_when_available(monkeypatch):
    text = "\u5c0f\u660e\u5728\u5317\u4eac\u5927\u5b66\u5b66\u4e60"

    class FakeHanLPModel:
        def __call__(self, tokens):
            assert tokens == list(text)
            return [
                ("\u5c0f\u660e", "PERSON", 0, 2),
                ("\u5317\u4eac\u5927\u5b66", "ORGANIZATION", 3, 7),
            ]

    monkeypatch.setattr(ner_module, "_get_hanlp_model", lambda: FakeHanLPModel())

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {
            "text": "\u5c0f\u660e",
            "label": "PER",
            "start": 0,
            "end": 2,
        },
        {
            "text": "\u5317\u4eac\u5927\u5b66",
            "label": "ORG",
            "start": 3,
            "end": 7,
        },
    ]


def test_recognize_entities_normalizes_location_labels(monkeypatch):
    text = "\u6211\u60f3\u53bb\u5317\u4eac\u65c5\u6e38"

    class FakeHanLPModel:
        def __call__(self, tokens):
            assert tokens == list(text)
            return [
                ("\u5317\u4eac", "ns", 3, 5),
            ]

    monkeypatch.setattr(ner_module, "_get_hanlp_model", lambda: FakeHanLPModel())

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {
            "text": "\u5317\u4eac",
            "label": "LOC",
            "start": 3,
            "end": 5,
        }
    ]


def test_recognize_entities_falls_back_to_rules_when_hanlp_is_unavailable(monkeypatch):
    text = "\u5c0f\u660e\u5728\u5317\u4eac\u5927\u5b66\u5b66\u4e60"

    monkeypatch.setattr(ner_module, "_get_hanlp_model", lambda: None)

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {
            "text": "\u5c0f\u660e",
            "label": "PER",
            "start": 0,
            "end": 2,
        },
        {
            "text": "\u5317\u4eac\u5927\u5b66",
            "label": "ORG",
            "start": 3,
            "end": 7,
        },
    ]


def test_recognize_entities_falls_back_to_rules_when_hanlp_prediction_fails(monkeypatch):
    text = "\u6211\u60f3\u53bb\u5317\u4eac\u65c5\u6e38"

    class BrokenHanLPModel:
        def __call__(self, _tokens):
            raise RuntimeError("model inference failed")

    monkeypatch.setattr(ner_module, "_get_hanlp_model", lambda: BrokenHanLPModel())

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {
            "text": "\u5317\u4eac",
            "label": "LOC",
            "start": 3,
            "end": 5,
        }
    ]
