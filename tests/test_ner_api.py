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
    assert any(entity["label"] in {"PER", "ORG", "LOC"} for entity in data["entities"])

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
