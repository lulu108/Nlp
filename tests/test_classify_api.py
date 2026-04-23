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


def test_classify_api_value_error(monkeypatch):
    def fake_classify(_text: str):
        raise ValueError("text cannot be empty")

    monkeypatch.setattr("backend.routes.classify.classify", fake_classify)

    app = create_app()
    client = app.test_client()

    resp = client.post("/api/classify", json={"text": "\u4eba\u5de5\u667a\u80fd\u63a8\u52a8\u79d1\u6280\u53d1\u5c55"})

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert data["message"] == "text cannot be empty"


def test_classify_api_file_not_found_error(monkeypatch):
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
    assert "model artifacts missing" in data["message"]


def test_classify_api_internal_server_error(monkeypatch):
    def fake_classify(_text: str):
        raise RuntimeError("boom")

    monkeypatch.setattr("backend.routes.classify.classify", fake_classify)

    app = create_app()
    client = app.test_client()

    resp = client.post("/api/classify", json={"text": "\u4eba\u5de5\u667a\u80fd\u63a8\u52a8\u79d1\u6280\u53d1\u5c55"})

    assert resp.status_code == 500
    data = resp.get_json()
    assert data["success"] is False
    assert data["message"] == "internal server error"
