from __future__ import annotations

from typing import Any

from flask import jsonify


def success_response(data: dict[str, Any] | None = None, status_code: int = 200):
	payload: dict[str, Any] = {"success": True}
	if data:
		payload.update(data)
	return jsonify(payload), status_code


def error_response(message: str, status_code: int = 400):
	payload = {
		"success": False,
		"message": message,
	}
	return jsonify(payload), status_code
