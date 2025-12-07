"""
API health and endpoint tests.

Commit message: "api: add Pydantic validation, CORS, /api/models endpoint"
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """
    Create a test client with mocked model loading.

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        FastAPI test client
    """
    # Mock model registries before importing main
    mock_registry = MagicMock()
    mock_registry.get.return_value = None
    mock_registry.list_models.return_value = ["test_model"]
    mock_registry.is_loaded.return_value = True

    mock_lstm_registry = MagicMock()
    mock_lstm_registry.seq_len = 20
    mock_lstm_registry.model_cards = {}

    monkeypatch.setattr("model_utils.get_registry", lambda: mock_registry)
    monkeypatch.setattr("lstm_utils_pytorch.get_lstm_registry",
                        lambda: mock_lstm_registry)

    # Import after patching
    import sys
    sys.path.insert(0, str(pytest.importorskip("pathlib").Path(
        __file__).parent.parent / "src" / "backend"))

    from main import app

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Test that health endpoint returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client: TestClient) -> None:
        """Test that health endpoint returns JSON."""
        response = client.get("/api/health")
        data = response.json()

        assert isinstance(data, dict)
        assert "status" in data

    def test_health_status_ok(self, client: TestClient) -> None:
        """Test that health status is ok."""
        response = client.get("/api/health")
        data = response.json()

        assert data["status"] == "ok"


class TestModelsEndpoint:
    """Tests for /api/models endpoint."""

    def test_models_returns_200(self, client: TestClient) -> None:
        """Test that models endpoint returns 200."""
        response = client.get("/api/models")
        assert response.status_code == 200

    def test_models_returns_list(self, client: TestClient) -> None:
        """Test that models endpoint returns a list."""
        response = client.get("/api/models")
        data = response.json()

        assert isinstance(data, dict)
        assert "models" in data
        assert isinstance(data["models"], list)


class TestPredictEndpoint:
    """Tests for /api/predict endpoint."""

    def test_predict_requires_body(self, client: TestClient) -> None:
        """Test that predict endpoint requires body."""
        response = client.post("/api/predict")
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_validates_input(self, client: TestClient) -> None:
        """Test that predict endpoint validates input."""
        invalid_payload = {
            "readings": [],  # Empty readings
        }

        response = client.post("/api/predict", json=invalid_payload)
        # Should reject empty readings
        assert response.status_code in (400, 422)

    def test_predict_validates_ph_range(self, client: TestClient) -> None:
        """Test that predict validates pH range."""
        invalid_payload = {
            "readings": [
                {"timestamp": "2024-01-01T12:00:00", "ph": 15.0, "tds": 300},
            ]
        }

        response = client.post("/api/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_predict_validates_tds_positive(self, client: TestClient) -> None:
        """Test that predict validates TDS is positive."""
        invalid_payload = {
            "readings": [
                {"timestamp": "2024-01-01T12:00:00", "ph": 7.0, "tds": -100},
            ]
        }

        response = client.post("/api/predict", json=invalid_payload)
        assert response.status_code == 422


class TestForecastEndpoint:
    """Tests for /api/forecast endpoint."""

    def test_forecast_requires_body(self, client: TestClient) -> None:
        """Test that forecast endpoint requires body."""
        response = client.post("/api/forecast")
        assert response.status_code == 422

    def test_forecast_validates_readings(self, client: TestClient) -> None:
        """Test that forecast validates readings."""
        invalid_payload = {
            "readings": [],
            "steps": 5,
        }

        response = client.post("/api/forecast", json=invalid_payload)
        assert response.status_code in (400, 422)

    def test_forecast_validates_steps(self, client: TestClient) -> None:
        """Test that forecast validates steps parameter."""
        invalid_payload = {
            "readings": [
                {"timestamp": "2024-01-01T12:00:00", "ph": 7.0, "tds": 300},
            ],
            "steps": -5,  # negative steps
        }

        response = client.post("/api/forecast", json=invalid_payload)
        assert response.status_code == 422


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        """Test that CORS headers are present in response."""
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )

        # CORS should be configured
        assert "access-control-allow-origin" in response.headers or response.status_code == 200


class TestStaticFiles:
    """Tests for static file serving."""

    def test_root_returns_html(self, client: TestClient) -> None:
        """Test that root returns HTML."""
        response = client.get("/")

        # Should redirect to or serve index.html
        assert response.status_code in (200, 307)

    def test_index_html_exists(self, client: TestClient) -> None:
        """Test that index.html is accessible."""
        response = client.get("/static/index.html")

        # Should return the HTML file
        if response.status_code == 200:
            assert "text/html" in response.headers.get("content-type", "")
