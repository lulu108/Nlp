from __future__ import annotations


def cluster_documents(
    documents: list[dict[str, str]], cluster_count: int | None = None
) -> list[dict[str, object]]:
    """Return deterministic 2D points for visualization demos."""
    n = len(documents)
    if n == 0:
        return []

    k = cluster_count if cluster_count is not None else 2
    k = max(1, min(k, n))

    points: list[dict[str, object]] = []
    for idx, doc in enumerate(documents):
        x = round((idx - (n - 1) / 2.0) * 0.73, 2)
        y = round((((idx % 2) * 2) - 1) * 0.61, 2)
        points.append(
            {
                "title": doc["title"],
                "x": x,
                "y": y,
                "cluster": idx % k,
            }
        )

    return points
