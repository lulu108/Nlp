from backend.app import create_app


def test_tokenize_api_success():
    app = create_app()
    client = app.test_client()

    resp = client.post("/api/tokenize", json={"text": "今天天气很好，我想去北京旅游。"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "tokens" in data
    assert isinstance(data["tokens"], list)


def test_tokenize_api_bad_request():
    app = create_app()
    client = app.test_client()

    resp = client.post("/api/tokenize", json={"text": "   "})

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "message" in data
