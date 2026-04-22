from __future__ import annotations

from flask import Blueprint, request

from backend.services.tokenize_service import tokenize
from backend.utils.response import error_response, success_response
from backend.utils.validators import validate_text_payload


tokenize_bp = Blueprint("tokenize", __name__)


@tokenize_bp.route("/api/tokenize", methods=["POST"])
def tokenize_route():
	payload = request.get_json(silent=True)
	is_valid, text, err = validate_text_payload(payload)
	if not is_valid:
		return error_response(err or "invalid request", 400)

	try:
		tokens = tokenize(text or "")
		return success_response({"tokens": tokens})
	except Exception:
		return error_response("internal server error", 500)
