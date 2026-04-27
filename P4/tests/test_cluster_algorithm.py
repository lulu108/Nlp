from __future__ import annotations

import pytest

from algorithms.cluster import cluster_documents


def _build_documents():
    return [
        {
            "title": "\u79d1\u6280A",
            "text": "\u82af\u7247\u7b97\u6cd5\u6a21\u578b\u4eba\u5de5\u667a\u80fd\u5e73\u53f0\u5347\u7ea7",
        },
        {
            "title": "\u79d1\u6280B",
            "text": "\u4eba\u5de5\u667a\u80fd\u82af\u7247\u6280\u672f\u7b97\u6cd5\u7cfb\u7edf\u521b\u65b0",
        },
        {
            "title": "\u4f53\u80b2A",
            "text": "\u7403\u961f\u8054\u8d5b\u6bd4\u8d5b\u51a0\u519b\u7403\u5458\u8fdb\u7403",
        },
        {
            "title": "\u4f53\u80b2B",
            "text": "\u7403\u5458\u5907\u6218\u8054\u8d5b\u6559\u7ec3\u9632\u5b88\u8fdb\u653b",
        },
    ]


def test_cluster_documents_returns_real_points():
    points = cluster_documents(_build_documents(), cluster_count=2)

    assert len(points) == 4

    for point in points:
        assert set(point.keys()) == {"title", "x", "y", "cluster"}
        assert isinstance(point["title"], str)
        assert isinstance(point["x"], float)
        assert isinstance(point["y"], float)
        assert isinstance(point["cluster"], int)

    cluster_map = {point["title"]: point["cluster"] for point in points}
    assert cluster_map["\u79d1\u6280A"] == cluster_map["\u79d1\u6280B"]
    assert cluster_map["\u4f53\u80b2A"] == cluster_map["\u4f53\u80b2B"]
    assert cluster_map["\u79d1\u6280A"] != cluster_map["\u4f53\u80b2A"]


def test_cluster_documents_invalid_cluster_count_raises_value_error():
    with pytest.raises(ValueError, match="cluster_count must be <= document count"):
        cluster_documents(_build_documents()[:2], cluster_count=3)


def test_cluster_documents_requires_at_least_two_documents():
    with pytest.raises(ValueError, match="at least two documents are required"):
        cluster_documents(_build_documents()[:1], cluster_count=2)
