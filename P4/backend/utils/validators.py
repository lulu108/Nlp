from __future__ import annotations

from typing import Any


def validate_text_payload(payload: Any) -> tuple[bool, str | None, str | None]:
	if not isinstance(payload, dict):
		return False, None, "request body must be a JSON object"

	text = payload.get("text")
	if not isinstance(text, str):
		return False, None, "text field is required"

	normalized = text.strip()
	if not normalized:
		return False, None, "text cannot be empty"

	return True, normalized, None


def validate_cluster_payload(
	payload: Any,
) -> tuple[bool, list[dict[str, str]] | None, int | None, str | None]:
	if not isinstance(payload, dict):
		return False, None, None, "request body must be a JSON object"

	documents = payload.get("documents")
	if not isinstance(documents, list):
		return False, None, None, "documents field is required"
	if len(documents) < 2:
		return False, None, None, "at least two documents are required"

	normalized_docs: list[dict[str, str]] = []
	for doc in documents:
		if not isinstance(doc, dict):
			return False, None, None, "each document must be an object"

		title = doc.get("title")
		text = doc.get("text")
		if not isinstance(title, str) or not title.strip():
			return False, None, None, "document title is required"
		if not isinstance(text, str) or not text.strip():
			return False, None, None, "document text is required"

		normalized_docs.append({"title": title.strip(), "text": text.strip()})

	cluster_count = payload.get("cluster_count")
	if cluster_count is not None:
		if not isinstance(cluster_count, int):
			return False, None, None, "cluster_count must be an integer"
		if cluster_count < 2:
			return False, None, None, "cluster_count must be >= 2"
		if cluster_count > len(normalized_docs):
			return False, None, None, "cluster_count must be <= document count"

	return True, normalized_docs, cluster_count, None
