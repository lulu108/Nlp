from __future__ import annotations

from flask import Blueprint, request

from backend.services.classify_service import classify
from backend.utils.response import error_response, success_response
from backend.utils.validators import validate_text_payload


classify_bp = Blueprint("classify", __name__)


@classify_bp.route("/api/classify", methods=["POST"])
def classify_route():
	payload = request.get_json(silent=True)
	is_valid, text, err = validate_text_payload(payload)
	if not is_valid:
		return error_response(err or "invalid request", 400)

	try:
		label, confidence = classify(text or "")
		return success_response({"label": label, "confidence": confidence})
	except ValueError as exc:
		return error_response(str(exc), 400)
	except FileNotFoundError as exc:
		return error_response(str(exc), 500)
	except Exception:
		return error_response("internal server error", 500)
