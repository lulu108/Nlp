from __future__ import annotations

from algorithms.cluster import cluster_documents


def cluster(documents: list[dict[str, str]], cluster_count: int | None = None):
	return cluster_documents(documents, cluster_count)
