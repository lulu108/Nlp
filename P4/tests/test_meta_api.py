from backend.app import create_app


def test_meta_api_exposes_runtime_identity():
	app = create_app()
	client = app.test_client()

	resp = client.get("/api/meta")

	assert resp.status_code == 200
	data = resp.get_json()
	assert data["success"] is True
	assert data["service"] == "P4 NLP backend"
	assert data["project_root"].endswith("P4")
	assert data["python_executable"]
	assert data["python_version"]
	assert isinstance(data["pid"], int)
	assert data["cwd"]
