"""
Tests for Generative Routes.

Tests the /api/generative_chat, /api/generate_report, and /api/sensor_health endpoints.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ========================
# Fixtures
# ========================


@pytest.fixture
def client():
    """Create test client with app."""
    # Import here to avoid issues with module loading
    import sys
    backend_path = Path(__file__).parent.parent / "src" / "backend"
    sys.path.insert(0, str(backend_path))

    from main import app
    return TestClient(app)


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data for testing."""
    now = datetime.now()
    data = []
    for i in range(100):
        timestamp = now - timedelta(minutes=5 * i)
        data.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "ph": 7.0 + (i % 10) * 0.1,
            "tds": 300 + (i % 20) * 10,
            "temp": 25 + (i % 5) * 0.5,
            "turbidity": 10 + (i % 10),
            "do": 7.5 + (i % 5) * 0.2,
            "hour": timestamp.hour,
            "hour_sin": 0.5,
            "hour_cos": 0.5,
            "is_dawn": 0,
            "ph_roll": 7.0,
            "tds_roll": 300,
            "delta_ph": 0.05,
            "delta_tds": 5,
            "ammonia_risk_raw": 175,
            "ammonia_risk": 0.5,
        })
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(sample_csv_data, tmp_path):
    """Create temporary data directory with sample CSV."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    csv_path = data_dir / "cleaned.csv"
    sample_csv_data.to_csv(csv_path, index=False)

    reports_dir = data_dir / "reports"
    reports_dir.mkdir()

    return data_dir


# ========================
# Generative Chat Tests
# ========================


class TestGenerativeChat:
    """Tests for /api/generative_chat endpoint."""

    def test_generative_chat_rule_based(self, client):
        """Test chat endpoint returns rule-based response."""
        response = client.post(
            "/api/generative_chat",
            json={
                "query": "What is the optimal pH for fish?",
                "context": None,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert "query" in data
        assert "reply" in data
        assert "source" in data
        assert data["source"] == "rule-based"
        assert len(data["reply"]) > 0

    def test_generative_chat_with_context(self, client):
        """Test chat with prediction context."""
        response = client.post(
            "/api/generative_chat",
            json={
                "query": "Why is my pH high?",
                "context": {
                    "last_prediction": {
                        "ph": 9.0,
                        "features": {"ph": 9.0, "tds": 400},
                    },
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "alkaline" in data["reply"].lower(
        ) or "high" in data["reply"].lower()

    def test_generative_chat_empty_query(self, client):
        """Test chat with empty query returns 422."""
        response = client.post(
            "/api/generative_chat",
            json={"query": "", "context": None},
        )

        assert response.status_code == 422

    def test_generative_chat_greeting(self, client):
        """Test chat with greeting."""
        response = client.post(
            "/api/generative_chat",
            json={"query": "Hello, can you help me?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "hello" in data["reply"].lower(
        ) or "help" in data["reply"].lower()

    def test_generative_chat_tds_query(self, client):
        """Test chat with TDS-related query."""
        response = client.post(
            "/api/generative_chat",
            json={
                "query": "What should my TDS levels be?",
                "context": {"last_prediction": {"tds": 350}},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "tds" in data["reply"].lower()

    def test_generative_chat_ammonia_query(self, client):
        """Test chat with ammonia query."""
        response = client.post(
            "/api/generative_chat",
            json={"query": "How do I reduce ammonia levels?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "ammonia" in data["reply"].lower(
        ) or "feeding" in data["reply"].lower()


# ========================
# Generate Report Tests
# ========================


class TestGenerateReport:
    """Tests for /api/generate_report endpoint."""

    def test_generate_report_no_data(self, client, monkeypatch, tmp_path):
        """Test report generation when data file missing."""
        # Patch the DATA_DIR to a non-existent location
        from routes import generative
        monkeypatch.setattr(generative, "DATA_DIR", tmp_path / "nonexistent")

        response = client.post(
            "/api/generate_report",
            json={"range": "daily"},
        )

        # Should return 404 because file doesn't exist
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_generate_report_success(self, client, temp_data_dir, monkeypatch):
        """Test successful report generation."""
        from routes import generative
        monkeypatch.setattr(generative, "DATA_DIR", temp_data_dir)
        monkeypatch.setattr(generative, "REPORTS_DIR",
                            temp_data_dir / "reports")

        response = client.post(
            "/api/generate_report",
            json={"range": "daily"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "attachment" in response.headers.get("content-disposition", "")
        assert len(response.content) > 0

    def test_generate_report_weekly(self, client, temp_data_dir, monkeypatch):
        """Test weekly report generation."""
        from routes import generative
        monkeypatch.setattr(generative, "DATA_DIR", temp_data_dir)
        monkeypatch.setattr(generative, "REPORTS_DIR",
                            temp_data_dir / "reports")

        response = client.post(
            "/api/generate_report",
            json={"range": "weekly"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"


# ========================
# Sensor Health Tests
# ========================


class TestSensorHealth:
    """Tests for /api/sensor_health endpoint."""

    def test_sensor_health_defaults(self, client, monkeypatch, tmp_path):
        """Test sensor health returns defaults when no data."""
        from routes import generative
        monkeypatch.setattr(generative, "DATA_DIR", tmp_path / "nonexistent")

        response = client.get("/api/sensor_health")

        assert response.status_code == 200
        data = response.json()

        assert "reliability_score" in data
        assert "drift_percent" in data
        assert "anomaly_rate" in data
        assert data["reliability_score"] >= 0
        assert data["reliability_score"] <= 100

    def test_sensor_health_with_data(self, client, temp_data_dir, monkeypatch):
        """Test sensor health with actual data."""
        from routes import generative
        monkeypatch.setattr(generative, "DATA_DIR", temp_data_dir)

        response = client.get("/api/sensor_health")

        assert response.status_code == 200
        data = response.json()

        assert "reliability_score" in data
        assert "drift_percent" in data
        assert "ph_drift" in data
        assert "tds_drift" in data
        assert "anomaly_rate" in data
        assert "recommendation" in data
        assert "drift_history" in data

    def test_sensor_health_with_lookback(self, client, temp_data_dir, monkeypatch):
        """Test sensor health with custom lookback."""
        from routes import generative
        monkeypatch.setattr(generative, "DATA_DIR", temp_data_dir)

        response = client.get("/api/sensor_health?lookback_minutes=120")

        assert response.status_code == 200
        data = response.json()
        assert "reliability_score" in data

    def test_sensor_health_with_device_id(self, client, temp_data_dir, monkeypatch):
        """Test sensor health with device filter."""
        from routes import generative
        monkeypatch.setattr(generative, "DATA_DIR", temp_data_dir)

        response = client.get("/api/sensor_health?device_id=sensor_1")

        assert response.status_code == 200
        data = response.json()
        # Device ID might not match, but endpoint should work
        assert "reliability_score" in data

    def test_sensor_health_drift_history_format(self, client, temp_data_dir, monkeypatch):
        """Test drift history has correct format for charts."""
        from routes import generative
        monkeypatch.setattr(generative, "DATA_DIR", temp_data_dir)

        response = client.get("/api/sensor_health")

        assert response.status_code == 200
        data = response.json()

        drift_history = data.get("drift_history")
        assert drift_history is not None
        assert "labels" in drift_history
        assert "phDrift" in drift_history
        assert "tdsDrift" in drift_history
        assert len(drift_history["labels"]) > 0


# ========================
# Integration Tests
# ========================


class TestIntegration:
    """Integration tests for multiple endpoints."""

    def test_chat_after_prediction_context(self, client):
        """Test chat can use prediction context."""
        # First make a prediction
        pred_response = client.post(
            "/api/predict",
            json={
                "timestamp": datetime.now().isoformat(),
                "ph": 8.8,
                "tds": 600,
                "device_id": "test_device",
            },
        )

        if pred_response.status_code == 200:
            prediction = pred_response.json()

            # Use prediction in chat context
            chat_response = client.post(
                "/api/generative_chat",
                json={
                    "query": "What does my pH level mean?",
                    "context": {"last_prediction": prediction},
                },
            )

            assert chat_response.status_code == 200
            chat_data = chat_response.json()
            assert len(chat_data["reply"]) > 0

    def test_endpoints_return_json(self, client):
        """Test all endpoints return valid JSON."""
        endpoints = [
            ("/api/health", "GET", None),
            ("/api/models", "GET", None),
            ("/api/sensor_health", "GET", None),
            ("/api/generative_chat", "POST", {"query": "test"}),
        ]

        for path, method, body in endpoints:
            if method == "GET":
                response = client.get(path)
            else:
                response = client.post(path, json=body)

            assert response.status_code in (
                200, 404, 422, 500), f"Endpoint {path} failed"
            # Should be JSON (except for PDF)
            if "pdf" not in response.headers.get("content-type", ""):
                try:
                    response.json()
                except Exception:
                    pytest.fail(f"Endpoint {path} did not return valid JSON")
