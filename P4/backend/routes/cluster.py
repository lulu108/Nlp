from __future__ import annotations

from flask import Blueprint, request

from backend.services.cluster_service import cluster
from backend.utils.response import error_response, success_response
from backend.utils.validators import validate_cluster_payload


cluster_bp = Blueprint("cluster", __name__)


@cluster_bp.route("/api/cluster", methods=["POST"])
def cluster_route():
	payload = request.get_json(silent=True)
	is_valid, documents, cluster_count, err = validate_cluster_payload(payload)
	if not is_valid:
		return error_response(err or "invalid request", 400)

	try:
		points = cluster(documents or [], cluster_count)
		return success_response({"points": points})
	except ValueError as exc:
		return error_response(str(exc), 400)
	except Exception:
		return error_response("internal server error", 500)
