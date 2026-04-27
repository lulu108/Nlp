from backend.app import create_app


def test_ner_api_success():
    app = create_app()
    client = app.test_client()

    resp = client.post("/api/ner", json={"text": "\u5c0f\u660e\u5728\u5317\u4eac\u5927\u5b66\u5b66\u4e60"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "entities" in data
    assert isinstance(data["entities"], list)
    assert any(
        entity["label"] in {"PER", "LOC", "ORG", "GPE", "FAC", "COMPANY", "INSTITUTION"}
        for entity in data["entities"]
    )

    for entity in data["entities"]:
        assert set(entity.keys()) == {"text", "label", "start", "end"}
        assert isinstance(entity["text"], str)
        assert isinstance(entity["label"], str)
        assert isinstance(entity["start"], int)
        assert isinstance(entity["end"], int)
        assert entity["start"] < entity["end"]


def test_ner_api_bad_request():
    app = create_app()
    client = app.test_client()

    resp = client.post("/api/ner", json={})

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "message" in data


def test_ner_api_uses_service_result(monkeypatch):
    def fake_ner(_text: str):
        return [
            {
                "text": "\u5c0f\u660e",
                "label": "PER",
                "start": 0,
                "end": 2,
            }
        ]

    monkeypatch.setattr("backend.routes.ner.ner", fake_ner)

    app = create_app()
    client = app.test_client()

    resp = client.post("/api/ner", json={"text": "\u5c0f\u660e\u5728\u5317\u4eac"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["entities"] == [
        {
            "text": "\u5c0f\u660e",
            "label": "PER",
            "start": 0,
            "end": 2,
        }
    ]


def test_meta_includes_ner_status():
    app = create_app()
    client = app.test_client()

    resp = client.get("/api/meta")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "ner_status" in data

    status = data["ner_status"]
    assert isinstance(status, dict)
    assert "tokenizer_model" in status
    assert "ner_model" in status
    assert "tokenizer_loaded" in status
    assert "ner_loaded" in status
    assert "last_used_path" in status
    assert "has_hanlp_load_error" in status
    assert "has_hanlp_predict_error" in status


def test_meta_ner_status_stable_when_hanlp_unavailable(monkeypatch):
    from algorithms import ner as ner_module

    monkeypatch.setattr(ner_module, "_get_hanlp_models", lambda: (None, None))

    app = create_app()
    client = app.test_client()

    ner_resp = client.post("/api/ner", json={"text": "\u9a6c\u4e91\u5728\u676d\u5dde"})
    assert ner_resp.status_code == 200

    meta_resp = client.get("/api/meta")
    assert meta_resp.status_code == 200

    meta_data = meta_resp.get_json()
    status = meta_data["ner_status"]
    assert status["last_used_path"] == "fallback"
    assert isinstance(status["tokenizer_loaded"], bool)
    assert isinstance(status["ner_loaded"], bool)
