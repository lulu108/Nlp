from backend.app import create_app


def test_classify_api_success(monkeypatch):
    def fake_classify(_text: str):
        return "科技", 0.8765

    monkeypatch.setattr("backend.routes.classify.classify", fake_classify)

    app = create_app()
    client = app.test_client()

    resp = client.post("/api/classify", json={"text": "北京有很多著名景点，适合旅游观光。"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["label"] == "科技"
    assert data["confidence"] == 0.8765
    assert "label" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_classify_api_bad_request():
    app = create_app()
    client = app.test_client()

    resp = client.post("/api/classify", json={"text": ""})

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "message" in data


def test_classify_api_server_error(monkeypatch):
    def fake_classify(_text: str):
        raise FileNotFoundError("model artifacts missing")

    monkeypatch.setattr("backend.routes.classify.classify", fake_classify)

    app = create_app()
    client = app.test_client()

    resp = client.post("/api/classify", json={"text": "人工智能推动科技发展"})

    assert resp.status_code == 500
    data = resp.get_json()
    assert data["success"] is False
    assert "message" in data
