from backend.app import create_app


def test_cluster_api_success():
    app = create_app()
    client = app.test_client()

    payload = {
        "cluster_count": 2,
        "documents": [
            {"title": "文本1", "text": "苹果公司发布了新产品。"},
            {"title": "文本2", "text": "人工智能推动了科技发展。"},
            {"title": "文本3", "text": "某球队赢得了比赛冠军。"},
        ],
    }
    resp = client.post("/api/cluster", json=payload)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "points" in data
    assert isinstance(data["points"], list)
    assert len(data["points"]) == 3


def test_cluster_api_bad_request():
    app = create_app()
    client = app.test_client()

    payload = {"documents": [{"title": "只有一条", "text": "不足聚类"}]}
    resp = client.post("/api/cluster", json=payload)

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "message" in data
