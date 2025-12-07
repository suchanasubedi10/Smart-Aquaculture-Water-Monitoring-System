"""
Tests for LSTM utilities (lstm_utils_pytorch.py).

Commit message: "backend: harden LSTM utils, add lazy-load and validation"
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


class TestValidateInputRows:
    """Tests for input validation."""

    def test_valid_input(self) -> None:
        """Test that valid input passes validation."""
        from lstm_utils_pytorch import validate_input_rows

        rows = [
            {"timestamp": "2024-01-01T12:00:00", "ph": 7.0, "tds": 300},
            {"timestamp": "2024-01-01T12:05:00", "ph": 7.1, "tds": 310},
        ]

        df = validate_input_rows(rows)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_empty_list_raises(self) -> None:
        """Test that empty list raises error."""
        from lstm_utils_pytorch import validate_input_rows

        with pytest.raises(ValueError, match="empty"):
            validate_input_rows([])

    def test_non_list_raises(self) -> None:
        """Test that non-list input raises error."""
        from lstm_utils_pytorch import validate_input_rows

        with pytest.raises(ValueError, match="list"):
            validate_input_rows("not a list")  # type: ignore

    def test_missing_keys_raises(self) -> None:
        """Test that missing required keys raises error."""
        from lstm_utils_pytorch import validate_input_rows

        rows = [
            {"timestamp": "2024-01-01T12:00:00", "ph": 7.0},  # missing tds
        ]

        with pytest.raises(ValueError, match="missing"):
            validate_input_rows(rows)

    def test_invalid_timestamp_raises(self) -> None:
        """Test that invalid timestamp raises error."""
        from lstm_utils_pytorch import validate_input_rows

        rows = [
            {"timestamp": "not-a-date", "ph": 7.0, "tds": 300},
        ]

        with pytest.raises(ValueError, match="timestamp"):
            validate_input_rows(rows)

    def test_invalid_ph_raises(self) -> None:
        """Test that invalid pH raises error."""
        from lstm_utils_pytorch import validate_input_rows

        rows = [
            {"timestamp": "2024-01-01T12:00:00", "ph": 15.0, "tds": 300},
        ]

        with pytest.raises(ValueError, match="pH"):
            validate_input_rows(rows)

    def test_negative_tds_raises(self) -> None:
        """Test that negative TDS raises error."""
        from lstm_utils_pytorch import validate_input_rows

        rows = [
            {"timestamp": "2024-01-01T12:00:00", "ph": 7.0, "tds": -100},
        ]

        with pytest.raises(ValueError, match="TDS"):
            validate_input_rows(rows)

    def test_sorts_by_timestamp(self) -> None:
        """Test that output is sorted by timestamp."""
        from lstm_utils_pytorch import validate_input_rows

        rows = [
            {"timestamp": "2024-01-01T12:10:00", "ph": 7.2, "tds": 320},
            {"timestamp": "2024-01-01T12:00:00", "ph": 7.0, "tds": 300},
            {"timestamp": "2024-01-01T12:05:00", "ph": 7.1, "tds": 310},
        ]

        df = validate_input_rows(rows)

        # Should be sorted ascending
        assert df.iloc[0]["ph"] == 7.0
        assert df.iloc[1]["ph"] == 7.1
        assert df.iloc[2]["ph"] == 7.2


class TestLstmForecast:
    """Tests for lstm_forecast function."""

    def test_returns_correct_structure(self, sample_readings_list: list[dict]) -> None:
        """Test that function returns correct structure."""
        from lstm_utils_pytorch import lstm_forecast

        result = lstm_forecast(sample_readings_list, steps=5)

        assert isinstance(result, dict)
        assert "ph" in result
        assert "tds" in result
        assert isinstance(result["ph"], list)
        assert isinstance(result["tds"], list)

    def test_empty_on_invalid_input(self) -> None:
        """Test that function returns empty on invalid input."""
        from lstm_utils_pytorch import lstm_forecast

        result = lstm_forecast([], steps=5)

        assert result["ph"] == []
        assert result["tds"] == []

    def test_handles_short_input(self) -> None:
        """Test that function handles input shorter than seq_len."""
        from lstm_utils_pytorch import lstm_forecast

        # Only 5 rows, less than typical seq_len of 20
        rows = [
            {"timestamp": f"2024-01-01T12:{i:02d}:00", "ph": 7.0, "tds": 300}
            for i in range(5)
        ]

        result = lstm_forecast(rows, steps=3)

        # Should still work (with padding)
        assert isinstance(result, dict)


class TestLSTMModelRegistry:
    """Tests for LSTM model registry."""

    def test_registry_singleton(self) -> None:
        """Test that registry is a singleton."""
        from lstm_utils_pytorch import get_lstm_registry

        reg1 = get_lstm_registry()
        reg2 = get_lstm_registry()

        assert reg1 is reg2

    def test_registry_has_seq_len(self) -> None:
        """Test that registry has seq_len property."""
        from lstm_utils_pytorch import get_lstm_registry

        reg = get_lstm_registry()
        assert isinstance(reg.seq_len, int)
        assert reg.seq_len > 0

    def test_registry_has_model_cards(self) -> None:
        """Test that registry has model_cards property."""
        from lstm_utils_pytorch import get_lstm_registry

        reg = get_lstm_registry()
        assert isinstance(reg.model_cards, dict)


class TestLSTMModel:
    """Tests for LSTM model class."""

    def test_model_instantiation(self) -> None:
        """Test that model can be instantiated."""
        from lstm_utils_pytorch import LSTMModel
        import torch

        model = LSTMModel(input_size=2, hidden_size=32, num_layers=1)
        assert isinstance(model, torch.nn.Module)

    def test_model_forward(self) -> None:
        """Test model forward pass."""
        from lstm_utils_pytorch import LSTMModel
        import torch

        model = LSTMModel(input_size=2, hidden_size=32, num_layers=1)
        x = torch.randn(8, 20, 2)  # batch=8, seq=20, features=2

        output = model(x)

        assert output.shape == (8,)
