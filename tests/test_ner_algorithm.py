from algorithms import ner as ner_module


def test_recognize_entities_uses_hanlp_when_available(monkeypatch):
    text = "\u5c0f\u660e\u5728\u5317\u4eac\u5927\u5b66\u5b66\u4e60"
    tokens = ["\u5c0f\u660e", "\u5728", "\u5317\u4eac\u5927\u5b66", "\u5b66\u4e60"]

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return tokens

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens == ["\u5c0f\u660e", "\u5728", "\u5317\u4eac\u5927\u5b66", "\u5b66\u4e60"]
            return [
                ("\u5c0f\u660e", "PERSON", 0, 1),
                ("\u5317\u4eac\u5927\u5b66", "ORGANIZATION", 2, 3),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

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

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return ["\u6211\u60f3", "\u53bb", "\u5317\u4eac", "\u65c5\u6e38"]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens == ["\u6211\u60f3", "\u53bb", "\u5317\u4eac", "\u65c5\u6e38"]
            return [
                ("\u5317\u4eac", "ns", 2, 3),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

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

    monkeypatch.setattr(ner_module, "_get_hanlp_models", lambda: (None, None))

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

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return ["\u6211\u60f3", "\u53bb", "\u5317\u4eac", "\u65c5\u6e38"]

    class BrokenHanLPNerModel:
        def __call__(self, _tokens):
            raise RuntimeError("model inference failed")

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), BrokenHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {
            "text": "\u5317\u4eac",
            "label": "LOC",
            "start": 3,
            "end": 5,
        }
    ]


def test_recognize_entities_keeps_full_entity_spans(monkeypatch):
    text = "\u9a6c\u4e91\u5728\u676d\u5dde\u521b\u529e\u963f\u91cc\u5df4\u5df4\u96c6\u56e2"

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return [
                "\u9a6c\u4e91",
                "\u5728",
                "\u676d\u5dde",
                "\u521b\u529e",
                "\u963f\u91cc\u5df4\u5df4\u96c6\u56e2",
            ]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens == [
                "\u9a6c\u4e91",
                "\u5728",
                "\u676d\u5dde",
                "\u521b\u529e",
                "\u963f\u91cc\u5df4\u5df4\u96c6\u56e2",
            ]
            return [
                ("\u9a6c\u4e91", "PERSON", 0, 1),
                ("\u676d\u5dde", "LOCATION", 2, 3),
                ("\u963f\u91cc\u5df4\u5df4\u96c6\u56e2", "ORGANIZATION", 4, 5),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {
            "text": "\u9a6c\u4e91",
            "label": "PER",
            "start": 0,
            "end": 2,
        },
        {
            "text": "\u676d\u5dde",
            "label": "LOC",
            "start": 3,
            "end": 5,
        },
        {
            "text": "\u963f\u91cc\u5df4\u5df4\u96c6\u56e2",
            "label": "ORG",
            "start": 7,
            "end": 13,
        },
    ]


def test_recognize_entities_accepts_char_level_spans_without_token_misread(monkeypatch):
    text = "\u9a6c\u4e91\u5728\u676d\u5dde\u521b\u529e\u4f01\u4e1a"

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return ["\u9a6c\u4e91", "\u5728", "\u676d\u5dde", "\u521b\u529e", "\u4f01\u4e1a"]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens == ["\u9a6c\u4e91", "\u5728", "\u676d\u5dde", "\u521b\u529e", "\u4f01\u4e1a"]
            # Simulate a model variant returning char-level offsets with entity text.
            return [
                ("\u9a6c\u4e91", "PERSON", 0, 2),
                ("\u676d\u5dde", "LOCATION", 3, 5),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {
            "text": "\u9a6c\u4e91",
            "label": "PER",
            "start": 0,
            "end": 2,
        },
        {
            "text": "\u676d\u5dde",
            "label": "LOC",
            "start": 3,
            "end": 5,
        },
    ]


def test_recognize_entities_handles_repeated_location_tokens(monkeypatch):
    text = "\u676d\u5dde\u548c\u676d\u5dde\u90fd\u5f88\u7f8e"

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return ["\u676d\u5dde", "\u548c", "\u676d\u5dde", "\u90fd", "\u5f88\u7f8e"]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens == ["\u676d\u5dde", "\u548c", "\u676d\u5dde", "\u90fd", "\u5f88\u7f8e"]
            return [
                ("\u676d\u5dde", "LOCATION", 0, 1),
                ("\u676d\u5dde", "LOCATION", 2, 3),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {"text": "\u676d\u5dde", "label": "LOC", "start": 0, "end": 2},
        {"text": "\u676d\u5dde", "label": "LOC", "start": 3, "end": 5},
    ]


def test_recognize_entities_handles_repeated_org_names(monkeypatch):
    text = "\u963f\u91cc\u5df4\u5df4\u96c6\u56e2\u548c\u963f\u91cc\u5df4\u5df4\u96c6\u56e2\u90fd\u53d1\u5e03\u4e86\u6d88\u606f"

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return [
                "\u963f\u91cc\u5df4\u5df4\u96c6\u56e2",
                "\u548c",
                "\u963f\u91cc\u5df4\u5df4\u96c6\u56e2",
                "\u90fd",
                "\u53d1\u5e03\u4e86",
                "\u6d88\u606f",
            ]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens[0] == "\u963f\u91cc\u5df4\u5df4\u96c6\u56e2"
            return [
                ("\u963f\u91cc\u5df4\u5df4\u96c6\u56e2", "ORG", 0, 1),
                ("\u963f\u91cc\u5df4\u5df4\u96c6\u56e2", "ORG", 2, 3),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {"text": "\u963f\u91cc\u5df4\u5df4\u96c6\u56e2", "label": "ORG", "start": 0, "end": 6},
        {"text": "\u963f\u91cc\u5df4\u5df4\u96c6\u56e2", "label": "ORG", "start": 7, "end": 13},
    ]


def test_recognize_entities_handles_tokens_with_punctuation(monkeypatch):
    text = "\u9a6c\u4e91\uff0c\u5728\u676d\u5dde\u3002"

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return ["\u9a6c\u4e91", "\uff0c", "\u5728", "\u676d\u5dde", "\u3002"]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens == ["\u9a6c\u4e91", "\uff0c", "\u5728", "\u676d\u5dde", "\u3002"]
            return [
                ("\u9a6c\u4e91", "PERSON", 0, 1),
                ("\u676d\u5dde", "LOCATION", 3, 4),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {"text": "\u9a6c\u4e91", "label": "PER", "start": 0, "end": 2},
        {"text": "\u676d\u5dde", "label": "LOC", "start": 4, "end": 6},
    ]


def test_recognize_entities_rejects_invalid_token_span_alignment(monkeypatch):
    text = "\u9a6c\u4e91\u5728\u676d\u5dde"

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            # The first token is intentionally unmatched to trigger invalid offset.
            return ["\u4e0d\u5b58\u5728\u7684token", "\u9a6c\u4e91", "\u5728", "\u676d\u5dde"]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens[0] == "\u4e0d\u5b58\u5728\u7684token"
            # This span covers an invalid token offset and should be dropped safely.
            return [
                ("\u9a6c\u4e91", "PERSON", 0, 2),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == []


def test_recognize_entities_preserves_selected_fine_grained_labels(monkeypatch):
    text = "北京市政府与北京大学合作"

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return ["北京市政府", "与", "北京大学", "合作"]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens == ["北京市政府", "与", "北京大学", "合作"]
            return [
                ("北京市政府", "GOVERNMENT", 0, 1),
                ("北京大学", "INSTITUTION", 2, 3),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {"text": "北京市政府", "label": "INSTITUTION", "start": 0, "end": 5},
        {"text": "北京大学", "label": "INSTITUTION", "start": 6, "end": 10},
    ]


def test_recognize_entities_maps_company_and_facility_labels(monkeypatch):
    text = "阿里巴巴总部位于杭州西湖景区"

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return ["阿里巴巴", "总部", "位于", "杭州西湖景区"]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens == ["阿里巴巴", "总部", "位于", "杭州西湖景区"]
            return [
                ("阿里巴巴", "CORPORATION", 0, 1),
                ("杭州西湖景区", "FAC", 3, 4),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {"text": "阿里巴巴", "label": "COMPANY", "start": 0, "end": 4},
        {"text": "杭州西湖景区", "label": "FAC", "start": 8, "end": 14},
    ]


def test_recognize_entities_unknown_label_falls_back_to_org(monkeypatch):
    text = "某组织发布公告"

    class FakeTokenizer:
        def __call__(self, input_text):
            assert input_text == text
            return ["某组织", "发布", "公告"]

    class FakeHanLPNerModel:
        def __call__(self, tokens):
            assert tokens == ["某组织", "发布", "公告"]
            return [
                ("某组织", "EVENT", 0, 1),
            ]

    monkeypatch.setattr(
        ner_module,
        "_get_hanlp_models",
        lambda: (FakeTokenizer(), FakeHanLPNerModel()),
    )

    entities = ner_module.recognize_entities(text)

    assert entities == [
        {"text": "某组织", "label": "ORG", "start": 0, "end": 3},
    ]
