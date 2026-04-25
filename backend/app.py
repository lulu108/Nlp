from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from backend.routes.classify import classify_bp
from backend.routes.cluster import cluster_bp
from backend.routes.ner import ner_bp
from backend.routes.tokenize import tokenize_bp
from backend.utils.response import success_response
from algorithms.ner import get_ner_runtime_status


def _read_bool_env(name: str, default: bool) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return int(value)
	except ValueError:
		return default


def create_app() -> Flask:
	app = Flask(__name__)
	CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

	app.register_blueprint(tokenize_bp)
	app.register_blueprint(ner_bp)
	app.register_blueprint(classify_bp)
	app.register_blueprint(cluster_bp)

	@app.get("/api/meta")
	def meta():
		return success_response(
			{
				"service": "P4 NLP backend",
				"project_root": str(ROOT_DIR),
				"python_executable": sys.executable,
				"python_version": platform.python_version(),
				"pid": os.getpid(),
				"cwd": os.getcwd(),
				"ner_status": get_ner_runtime_status(),
			}
		)

	return app


app = create_app()


if __name__ == "__main__":
	host = os.getenv("FLASK_HOST", DEFAULT_HOST)
	port = _read_int_env("PORT", DEFAULT_PORT)
	debug = _read_bool_env("FLASK_DEBUG", False)
	use_reloader = _read_bool_env("FLASK_USE_RELOADER", debug)

	print(f"[P4 backend] root={ROOT_DIR}")
	print(f"[P4 backend] python={sys.executable}")
	print(f"[P4 backend] pid={os.getpid()}")
	print(f"[P4 backend] listening on http://{host}:{port}")
	print(f"[P4 backend] debug={debug}, reloader={use_reloader}")

	app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)
