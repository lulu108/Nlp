from __future__ import annotations

import pytest

from algorithms.cluster import cluster_documents
from algorithms.tokenizer import tokenize_text


def sample_documents() -> list[dict[str, str]]:
    return [
        {
            "title": "科技文档A",
            "text": "人工智能模型和芯片算法提升计算平台能力。",
        },
        {
            "title": "科技文档B",
            "text": "机器学习系统分析文本数据并输出智能结果。",
        },
        {
            "title": "体育文档A",
            "text": "球队完成训练后参加联赛比赛并取得胜利。",
        },
        {
            "title": "体育文档B",
            "text": "教练安排球员备战冠军决赛和足球比赛。",
        },
    ]


def test_tokenize_text_returns_tokens_for_normal_text():
    tokens = tokenize_text("北京大学学习自然语言处理")

    assert isinstance(tokens, list)
    assert tokens
    assert all(isinstance(token, str) and token for token in tokens)


@pytest.mark.parametrize("value", ["", "   "])
def test_tokenize_text_returns_empty_list_for_blank_text(value):
    assert tokenize_text(value) == []


def test_cluster_documents_returns_visualization_points():
    documents = sample_documents()

    points = cluster_documents(documents, cluster_count=2)

    assert isinstance(points, list)
    assert len(points) == len(documents)
    for point in points:
        assert {"title", "x", "y", "cluster"} == set(point)
        assert isinstance(point["title"], str)
        assert isinstance(point["x"], float)
        assert isinstance(point["y"], float)
        assert isinstance(point["cluster"], int)
        assert 0 <= point["cluster"] < 2


@pytest.mark.parametrize(
    ("documents", "cluster_count", "message"),
    [
        ([{"title": "单篇", "text": "只有一篇文档"}], 2, "at least two documents"),
        (
            [{"text": "缺少标题"}, {"title": "正常标题", "text": "另一篇文档"}],
            2,
            "missing title",
        ),
        (
            [{"title": "缺少正文"}, {"title": "正常标题", "text": "另一篇文档"}],
            2,
            "missing text",
        ),
        (sample_documents(), 1, "cluster_count must be >= 2"),
        (sample_documents(), 5, "cluster_count must be <= document count"),
    ],
)
def test_cluster_documents_rejects_invalid_inputs(documents, cluster_count, message):
    with pytest.raises(ValueError, match=message):
        cluster_documents(documents, cluster_count=cluster_count)
