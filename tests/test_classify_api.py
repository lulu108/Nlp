from backend.app import create_app


def test_classify_api_success():
    app = create_app()
    client = app.test_client()

    resp = client.post("/api/classify", json={"text": "北京有很多著名景点，适合旅游观光。"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
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
