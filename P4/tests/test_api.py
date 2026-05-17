from __future__ import annotations

import pytest


def sample_documents() -> list[dict[str, str]]:
    return [
        {
            "title": "科技新闻A",
            "text": "人工智能芯片算法平台发布新版本，模型训练效率提升。",
        },
        {
            "title": "科技新闻B",
            "text": "智能计算系统使用机器学习算法分析数据和文本。",
        },
        {
            "title": "体育新闻A",
            "text": "球队在联赛决赛中进球获胜，教练称赞球员表现。",
        },
        {
            "title": "体育新闻B",
            "text": "冠军球员完成训练后备战下一场足球比赛。",
        },
    ]


def assert_error_response(response, expected_status: int = 400) -> None:
    assert response.status_code == expected_status
    data = response.get_json()
    assert data["success"] is False
    assert isinstance(data["message"], str)
    assert data["message"]


def test_meta_api(client):
    response = client.get("/api/meta")

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["service"] == "P4 NLP backend"
    assert data["project_root"].endswith("P4")
    assert "ner_status" in data


def test_tokenize_api_success(client):
    response = client.post("/api/tokenize", json={"text": "北京大学学习自然语言处理"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert isinstance(data["tokens"], list)
    assert data["tokens"]
    assert all(isinstance(token, str) for token in data["tokens"])


@pytest.mark.parametrize("payload", [{"text": ""}, {}])
def test_tokenize_api_rejects_invalid_text(client, payload):
    response = client.post("/api/tokenize", json=payload)

    assert_error_response(response)


def test_ner_api_success_returns_entity_list(client, monkeypatch):
    from algorithms import ner as ner_algorithm

    monkeypatch.setattr(ner_algorithm, "_get_hanlp_models", lambda: (None, None))

    response = client.post("/api/ner", json={"text": "小明在北京大学学习。"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert isinstance(data["entities"], list)
    for entity in data["entities"]:
        assert {"text", "label", "start", "end"}.issubset(entity)


@pytest.mark.parametrize("payload", [{"text": ""}, {}])
def test_ner_api_rejects_invalid_text(client, payload):
    response = client.post("/api/ner", json=payload)

    assert_error_response(response)


def test_classify_api_success_when_model_files_exist(
    client, classifier_model_files_exist: bool
):
    if not classifier_model_files_exist:
        pytest.skip("classifier model artifacts are not available")

    response = client.post("/api/classify", json={"text": "人工智能算法推动产业发展"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert isinstance(data["label"], str)
    assert data["label"]
    assert isinstance(data["confidence"], (float, int))
    assert 0 <= data["confidence"] <= 1


@pytest.mark.parametrize("payload", [{"text": ""}, {}])
def test_classify_api_rejects_invalid_text(client, payload):
    response = client.post("/api/classify", json=payload)

    assert_error_response(response)


def test_cluster_api_success(client):
    response = client.post(
        "/api/cluster",
        json={"documents": sample_documents(), "cluster_count": 2},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert isinstance(data["points"], list)
    assert len(data["points"]) == len(sample_documents())
    for point in data["points"]:
        assert {"title", "x", "y", "cluster"} == set(point)
        assert isinstance(point["title"], str)
        assert isinstance(point["x"], (float, int))
        assert isinstance(point["y"], (float, int))
        assert isinstance(point["cluster"], int)


@pytest.mark.parametrize(
    "payload",
    [
        {"documents": [{"title": "单篇", "text": "只有一篇文档"}], "cluster_count": 2},
        {
            "documents": [
                {"text": "缺少标题"},
                {"title": "正常标题", "text": "另一篇文档"},
            ],
            "cluster_count": 2,
        },
        {
            "documents": [
                {"title": "缺少正文"},
                {"title": "正常标题", "text": "另一篇文档"},
            ],
            "cluster_count": 2,
        },
        {"documents": sample_documents(), "cluster_count": 1},
        {"documents": sample_documents(), "cluster_count": 5},
    ],
)
def test_cluster_api_rejects_invalid_payloads(client, payload):
    response = client.post("/api/cluster", json=payload)

    assert_error_response(response)
