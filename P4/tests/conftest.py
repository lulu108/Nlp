from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def app():
    from backend import app as app_module

    if hasattr(app_module, "create_app"):
        flask_app = app_module.create_app()
    else:
        flask_app = app_module.app

    flask_app.config.update(TESTING=True)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def classifier_model_files_exist() -> bool:
    model_dir = PROJECT_ROOT / "models" / "classifier"
    return (
        (model_dir / "tfidf_vectorizer.pkl").exists()
        and (model_dir / "svm_model.pkl").exists()
    )
