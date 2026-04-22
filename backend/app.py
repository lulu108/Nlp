from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from backend.routes.classify import classify_bp
from backend.routes.cluster import cluster_bp
from backend.routes.ner import ner_bp
from backend.routes.tokenize import tokenize_bp


def create_app() -> Flask:
	app = Flask(__name__)

	app.register_blueprint(tokenize_bp)
	app.register_blueprint(ner_bp)
	app.register_blueprint(classify_bp)
	app.register_blueprint(cluster_bp)

	return app


app = create_app()


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
