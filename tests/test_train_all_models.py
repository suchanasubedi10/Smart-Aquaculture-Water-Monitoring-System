"""
Tests for training pipeline (train_all_models.py).

Commit message: "scripts: refactor training pipeline, add metrics and model cards"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestFeaturize:
    """Tests for the featurize function."""

    def test_featurize_adds_time_features(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that featurize adds time-based features."""
        from train_all_models import featurize

        df = featurize(sample_dataframe)

        assert "hour" in df.columns
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "is_dawn" in df.columns

    def test_featurize_adds_rolling_features(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that featurize adds rolling window features."""
        from train_all_models import featurize

        df = featurize(sample_dataframe)

        assert "ph_roll" in df.columns
        assert "tds_roll" in df.columns

    def test_featurize_adds_delta_features(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that featurize adds delta features."""
        from train_all_models import featurize

        df = featurize(sample_dataframe)

        assert "delta_ph" in df.columns
        assert "delta_tds" in df.columns

    def test_featurize_adds_ammonia_risk(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that featurize adds ammonia risk feature."""
        from train_all_models import featurize

        df = featurize(sample_dataframe)

        assert "ammonia_risk_raw" in df.columns

    def test_hour_sin_cos_in_valid_range(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that sin/cos features are in [-1, 1]."""
        from train_all_models import featurize

        df = featurize(sample_dataframe)

        assert df["hour_sin"].min() >= -1
        assert df["hour_sin"].max() <= 1
        assert df["hour_cos"].min() >= -1
        assert df["hour_cos"].max() <= 1

    def test_is_dawn_binary(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that is_dawn is binary."""
        from train_all_models import featurize

        df = featurize(sample_dataframe)

        assert set(df["is_dawn"].unique()).issubset({0, 1})


class TestCleanData:
    """Tests for data cleaning function."""

    def test_clean_removes_negative_ph(self) -> None:
        """Test that cleaning removes negative pH values."""
        from train_all_models import clean_data

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "ph": [7.0, -1.0, 7.5, 8.0, 7.2, 7.1, 7.3, 7.4, 7.0, 7.1],
            "tds": [300] * 10,
            "temp": [25] * 10,
            "do": [7] * 10,
            "turbidity": [50] * 10,
        })

        df_clean = clean_data(df)
        assert (df_clean["ph"] > 0).all()

    def test_clean_removes_negative_tds(self) -> None:
        """Test that cleaning removes negative TDS values."""
        from train_all_models import clean_data

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "ph": [7.0] * 10,
            "tds": [300, -50, 310, 320, 290, 300, 305, 295, 300, 310],
            "temp": [25] * 10,
            "do": [7] * 10,
            "turbidity": [50] * 10,
        })

        df_clean = clean_data(df)
        assert (df_clean["tds"] >= 0).all()


class TestLoadData:
    """Tests for data loading function."""

    def test_load_data_returns_dataframe(self, tmp_path: Path) -> None:
        """Test that load_data returns a DataFrame."""
        from train_all_models import load_data

        # Create test CSV
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "ph": [7.0] * 10,
            "tds": [300] * 10,
        })
        df.to_csv(csv_path, index=False)

        result = load_data(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_load_data_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Test that load_data raises on missing file."""
        from train_all_models import load_data

        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nonexistent.csv")

    def test_load_data_raises_on_missing_columns(self, tmp_path: Path) -> None:
        """Test that load_data raises on missing required columns."""
        from train_all_models import load_data

        csv_path = tmp_path / "incomplete.csv"
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "ph": [7.0] * 5,
            # Missing 'tds' column
        })
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(csv_path)


class TestModelCard:
    """Tests for model card creation."""

    def test_create_model_card_has_required_fields(self) -> None:
        """Test that model card has all required fields."""
        from train_all_models import create_model_card

        card = create_model_card(
            name="test_model",
            model_type="RandomForestRegressor",
            args={"n_estimators": 100},
            feature_list=["ph", "tds"],
            metrics={"rmse": 0.5},
        )

        assert "name" in card
        assert "version" in card
        assert "created_at" in card
        assert "model_type" in card
        assert "training_args" in card
        assert "feature_list" in card
        assert "metrics_summary" in card

    def test_create_model_card_version_is_uuid(self) -> None:
        """Test that version is a valid UUID string."""
        from train_all_models import create_model_card
        import uuid

        card = create_model_card(
            name="test",
            model_type="RF",
            args={},
            feature_list=[],
        )

        # Should not raise
        uuid.UUID(card["version"])
