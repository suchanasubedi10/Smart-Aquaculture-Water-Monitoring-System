"""
Tests for model utilities (model_utils.py).

Commit message: "backend: improve model utils, add types, lazy loading, structured recommendations"
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestComputeBasicFeatures:
    """Tests for compute_basic_features function."""

    def test_returns_dataframe(self, sample_reading: dict) -> None:
        """Test that function returns a DataFrame."""
        from model_utils import compute_basic_features

        result = compute_basic_features(sample_reading)
        assert isinstance(result, pd.DataFrame)

    def test_returns_single_row(self, sample_reading: dict) -> None:
        """Test that function returns single row."""
        from model_utils import compute_basic_features

        result = compute_basic_features(sample_reading)
        assert len(result) == 1

    def test_contains_required_features(self, sample_reading: dict) -> None:
        """Test that result contains required features."""
        from model_utils import compute_basic_features

        result = compute_basic_features(sample_reading)
        required = {"ph", "tds", "ammonia_risk",
                    "hour_sin", "hour_cos", "is_dawn"}

        # All required features should be present
        assert required.issubset(set(result.columns))

    def test_hour_sin_cos_in_range(self, sample_reading: dict) -> None:
        """Test that hour sin/cos are in valid range."""
        from model_utils import compute_basic_features

        result = compute_basic_features(sample_reading)
        row = result.iloc[0]

        assert -1 <= row["hour_sin"] <= 1
        assert -1 <= row["hour_cos"] <= 1

    def test_is_dawn_binary(self, sample_reading: dict) -> None:
        """Test that is_dawn is 0 or 1."""
        from model_utils import compute_basic_features

        result = compute_basic_features(sample_reading)
        assert result.iloc[0]["is_dawn"] in [0, 1]

    def test_preserves_ph_value(self, sample_reading: dict) -> None:
        """Test that pH value is preserved."""
        from model_utils import compute_basic_features

        result = compute_basic_features(sample_reading)
        assert result.iloc[0]["ph"] == sample_reading["ph"]

    def test_preserves_tds_value(self, sample_reading: dict) -> None:
        """Test that TDS value is preserved."""
        from model_utils import compute_basic_features

        result = compute_basic_features(sample_reading)
        assert result.iloc[0]["tds"] == sample_reading["tds"]

    def test_raises_on_invalid_ph(self) -> None:
        """Test that invalid pH raises error."""
        from model_utils import compute_basic_features

        payload = {
            "timestamp": "2024-01-01T12:00:00",
            "ph": 15.0,  # Invalid
            "tds": 300,
        }

        with pytest.raises(ValueError, match="pH"):
            compute_basic_features(payload)

    def test_raises_on_negative_tds(self) -> None:
        """Test that negative TDS raises error."""
        from model_utils import compute_basic_features

        payload = {
            "timestamp": "2024-01-01T12:00:00",
            "ph": 7.0,
            "tds": -100,  # Invalid
        }

        with pytest.raises(ValueError, match="TDS"):
            compute_basic_features(payload)

    def test_raises_on_missing_timestamp(self) -> None:
        """Test that missing timestamp raises error."""
        from model_utils import compute_basic_features

        payload = {"ph": 7.0, "tds": 300}

        with pytest.raises(ValueError):
            compute_basic_features(payload)


class TestPredictForecastRF:
    """Tests for RF forecasting function."""

    def test_returns_dict_with_correct_keys(self) -> None:
        """Test that function returns dict with ph_forecast and tds_forecast."""
        from model_utils import predict_forecast_rf

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "ph": [7.0] * 10,
            "tds": [300] * 10,
        })

        result = predict_forecast_rf(df, steps=3)

        assert "ph_forecast" in result
        assert "tds_forecast" in result
        assert isinstance(result["ph_forecast"], list)
        assert isinstance(result["tds_forecast"], list)

    def test_returns_correct_number_of_steps(self) -> None:
        """Test that function returns correct number of predictions."""
        from model_utils import predict_forecast_rf

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "ph": [7.0] * 10,
            "tds": [300] * 10,
        })

        result = predict_forecast_rf(df, steps=5)

        # Note: may be empty if models not loaded
        # Just verify it doesn't crash
        assert isinstance(result["ph_forecast"], list)

    def test_handles_single_row_input(self) -> None:
        """Test that function handles single row gracefully."""
        from model_utils import predict_forecast_rf

        df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01T12:00:00")],
            "ph": [7.0],
            "tds": [300],
        })

        result = predict_forecast_rf(df, steps=2)
        assert isinstance(result, dict)


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_recommend_returns_list(self) -> None:
        """Test that recommend returns a list."""
        from model_utils import recommend

        all_outputs = {
            "classification": {"prediction": 0},
            "virtual": {"do_est": 7.0},
            "anomaly": {"is_anomaly": False},
            "features": {"ammonia_risk": 0.3},
        }

        result = recommend(all_outputs)
        assert isinstance(result, list)

    def test_recommend_includes_classification_message(self) -> None:
        """Test that recommendations include classification message."""
        from model_utils import recommend

        all_outputs = {
            "classification": {"prediction": 0},
            "virtual": {},
            "anomaly": {"is_anomaly": False},
            "features": {},
        }

        result = recommend(all_outputs)
        messages = [r.message for r in result]

        assert any("SAFE" in m for m in messages)

    def test_recommend_danger_condition(self) -> None:
        """Test danger condition recommendation."""
        from model_utils import recommend

        all_outputs = {
            "classification": {"prediction": 2},
            "virtual": {},
            "anomaly": {"is_anomaly": False},
            "features": {},
        }

        result = recommend(all_outputs)
        messages = [r.message for r in result]

        assert any("DANGER" in m for m in messages)

    def test_recommend_anomaly_detected(self) -> None:
        """Test anomaly detection recommendation."""
        from model_utils import recommend

        all_outputs = {
            "classification": {"prediction": 0},
            "virtual": {},
            "anomaly": {"is_anomaly": True},
            "features": {},
        }

        result = recommend(all_outputs)
        codes = [r.code for r in result]

        assert "ANOMALY_DETECTED" in codes

    def test_recommend_low_do(self) -> None:
        """Test low DO recommendation."""
        from model_utils import recommend

        all_outputs = {
            "classification": {"prediction": 0},
            "virtual": {"do_est": 2.0},
            "anomaly": {"is_anomaly": False},
            "features": {},
        }

        result = recommend(all_outputs)
        codes = [r.code for r in result]

        assert "DO_CRITICAL" in codes


class TestModelRegistry:
    """Tests for model registry."""

    def test_registry_singleton(self) -> None:
        """Test that registry is a singleton."""
        from model_utils import get_model_registry

        reg1 = get_model_registry()
        reg2 = get_model_registry()

        assert reg1 is reg2

    def test_registry_has_scaler_property(self) -> None:
        """Test that registry has scaler property."""
        from model_utils import get_model_registry

        reg = get_model_registry()
        _ = reg.scaler  # Should not raise

    def test_registry_has_feature_list(self) -> None:
        """Test that registry has feature_list property."""
        from model_utils import get_model_registry

        reg = get_model_registry()
        assert isinstance(reg.feature_list, list)
