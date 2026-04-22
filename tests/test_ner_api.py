from backend.app import create_app


def test_ner_api_success():
    app = create_app()
    client = app.test_client()

    resp = client.post("/api/ner", json={"text": "小明在北京大学学习。"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "entities" in data
    assert isinstance(data["entities"], list)


def test_ner_api_bad_request():
    app = create_app()
    client = app.test_client()

    resp = client.post("/api/ner", json={})

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "message" in data
