from backend.app import create_app


def _build_documents():
    return [
        {
            "title": "\u79d1\u6280\u6587\u672c1",
            "text": "\u82af\u7247\u7b97\u6cd5\u6a21\u578b\u4eba\u5de5\u667a\u80fd\u5e73\u53f0\u6301\u7eed\u5347\u7ea7",
        },
        {
            "title": "\u79d1\u6280\u6587\u672c2",
            "text": "\u4eba\u5de5\u667a\u80fd\u82af\u7247\u7cfb\u7edf\u4e0e\u7b97\u6cd5\u6280\u672f\u5feb\u901f\u53d1\u5c55",
        },
        {
            "title": "\u4f53\u80b2\u6587\u672c1",
            "text": "\u7403\u961f\u8054\u8d5b\u51a0\u519b\u6bd4\u8d5b\u8fdb\u7403\u6559\u7ec3\u7403\u5458\u8868\u73b0\u51fa\u8272",
        },
        {
            "title": "\u4f53\u80b2\u6587\u672c2",
            "text": "\u7403\u5458\u5907\u6218\u8054\u8d5b\u6bd4\u8d5b\u9632\u5b88\u8fdb\u653b\u5e2e\u52a9\u7403\u961f\u53d6\u80dc",
        },
    ]


def test_cluster_api_success():
    app = create_app()
    client = app.test_client()

    payload = {
        "cluster_count": 2,
        "documents": _build_documents(),
    }
    resp = client.post("/api/cluster", json=payload)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "points" in data
    assert isinstance(data["points"], list)
    assert len(data["points"]) == 4

    for point in data["points"]:
        assert set(point.keys()) == {"title", "x", "y", "cluster"}
        assert isinstance(point["title"], str)
        assert isinstance(point["x"], float)
        assert isinstance(point["y"], float)
        assert isinstance(point["cluster"], int)


def test_cluster_api_invalid_cluster_count():
    app = create_app()
    client = app.test_client()

    payload = {
        "cluster_count": 5,
        "documents": _build_documents()[:3],
    }
    resp = client.post("/api/cluster", json=payload)

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "cluster_count must be <= document count" in data["message"]


def test_cluster_api_bad_request():
    app = create_app()
    client = app.test_client()

    payload = {
        "documents": [
            {
                "title": "\u53ea\u6709\u4e00\u6761",
                "text": "\u6587\u6863\u6570\u91cf\u4e0d\u8db3",
            }
        ]
    }
    resp = client.post("/api/cluster", json=payload)

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "at least two documents are required" in data["message"]
