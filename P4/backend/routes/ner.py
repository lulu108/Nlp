from __future__ import annotations

from flask import Blueprint, request

from backend.services.ner_service import ner
from backend.utils.response import error_response, success_response
from backend.utils.validators import validate_text_payload


ner_bp = Blueprint("ner", __name__)


@ner_bp.route("/api/ner", methods=["POST"])
def ner_route():
	payload = request.get_json(silent=True)
	is_valid, text, err = validate_text_payload(payload)
	if not is_valid:
		return error_response(err or "invalid request", 400)

	try:
		entities = ner(text or "")
		return success_response({"entities": entities})
	except Exception:
		return error_response("internal server error", 500)
