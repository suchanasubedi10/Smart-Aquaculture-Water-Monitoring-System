"""
Tests for evaluate_classifier.py.

Commit message: "scripts: refactor evaluate_classifier with CLI and metrics export"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path for imports
# This is also done in conftest.py but we do it here for IDE support
_scripts_path = str(Path(__file__).parent.parent / "scripts")
if _scripts_path not in sys.path:
    sys.path.insert(0, _scripts_path)

# Import after path is set up
from evaluate_classifier import (
    label_row,
    compute_evaluation_metrics,
    load_classifier,
    load_and_validate_data,
)


class TestLabelRow:
    """Tests for the rule-based labeling function."""

    def test_safe_normal_values(self) -> None:
        """Test that normal values are labeled Safe."""
        row = pd.Series({"ph": 7.0, "tds": 300, "temp": 25, "turbidity": 20})
        assert label_row(row) == 0  # Safe

    def test_danger_low_ph(self) -> None:
        """Test that low pH is labeled Danger."""
        row = pd.Series({"ph": 5.0, "tds": 300, "temp": 25, "turbidity": 20})
        assert label_row(row) == 2  # Danger

    def test_danger_high_ph(self) -> None:
        """Test that high pH is labeled Danger."""
        row = pd.Series({"ph": 10.0, "tds": 300, "temp": 25, "turbidity": 20})
        assert label_row(row) == 2  # Danger

    def test_danger_high_tds(self) -> None:
        """Test that very high TDS is labeled Danger."""
        row = pd.Series({"ph": 7.0, "tds": 1500, "temp": 25, "turbidity": 20})
        assert label_row(row) == 2  # Danger

    def test_danger_low_temp(self) -> None:
        """Test that low temperature is labeled Danger."""
        row = pd.Series({"ph": 7.0, "tds": 300, "temp": 10, "turbidity": 20})
        assert label_row(row) == 2  # Danger

    def test_danger_high_temp(self) -> None:
        """Test that high temperature is labeled Danger."""
        row = pd.Series({"ph": 7.0, "tds": 300, "temp": 40, "turbidity": 20})
        assert label_row(row) == 2  # Danger

    def test_danger_high_turbidity(self) -> None:
        """Test that very high turbidity is labeled Danger."""
        row = pd.Series({"ph": 7.0, "tds": 300, "temp": 25, "turbidity": 200})
        assert label_row(row) == 2  # Danger

    def test_stress_marginal_ph_low(self) -> None:
        """Test that marginally low pH is labeled Stress."""
        row = pd.Series({"ph": 6.2, "tds": 300, "temp": 25, "turbidity": 20})
        assert label_row(row) == 1  # Stress

    def test_stress_marginal_ph_high(self) -> None:
        """Test that marginally high pH is labeled Stress."""
        row = pd.Series({"ph": 8.7, "tds": 300, "temp": 25, "turbidity": 20})
        assert label_row(row) == 1  # Stress

    def test_stress_high_tds(self) -> None:
        """Test that high TDS (but not critical) is labeled Stress."""
        row = pd.Series({"ph": 7.0, "tds": 700, "temp": 25, "turbidity": 20})
        assert label_row(row) == 1  # Stress

    def test_stress_moderate_turbidity(self) -> None:
        """Test that moderate turbidity is labeled Stress."""
        row = pd.Series({"ph": 7.0, "tds": 300, "temp": 25, "turbidity": 80})
        assert label_row(row) == 1  # Stress

    def test_missing_temp_uses_default(self) -> None:
        """Test that missing temp uses default value."""
        # No temp column - should use default of 25 and be Safe
        row = pd.Series({"ph": 7.0, "tds": 300, "turbidity": 20})
        assert label_row(row) == 0  # Safe

    def test_missing_turbidity_uses_default(self) -> None:
        """Test that missing turbidity uses default value."""
        # No turbidity column - should use default of 0 and be Safe
        row = pd.Series({"ph": 7.0, "tds": 300, "temp": 25})
        assert label_row(row) == 0  # Safe


class TestComputeEvaluationMetrics:
    """Tests for metrics computation."""

    def test_perfect_predictions(self) -> None:
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_evaluation_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["weighted_f1"] == 1.0
        assert "confusion_matrix" in metrics
        assert "per_class" in metrics

    def test_confusion_matrix_shape(self) -> None:
        """Test that confusion matrix has correct shape."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 0, 2, 1, 1, 2])

        metrics = compute_evaluation_metrics(y_true, y_pred)
        cm = np.array(metrics["confusion_matrix"])

        assert cm.shape == (3, 3)

    def test_per_class_metrics(self) -> None:
        """Test that per-class metrics are computed."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        metrics = compute_evaluation_metrics(y_true, y_pred)

        assert "Safe" in metrics["per_class"]
        assert "Stress" in metrics["per_class"]
        assert "Danger" in metrics["per_class"]

        for label in ["Safe", "Stress", "Danger"]:
            assert "precision" in metrics["per_class"][label]
            assert "recall" in metrics["per_class"][label]
            assert "f1" in metrics["per_class"][label]
            assert "support" in metrics["per_class"][label]

    def test_class_distribution(self) -> None:
        """Test that class distribution is computed."""
        y_true = np.array([0, 0, 0, 1, 2])
        y_pred = np.array([0, 0, 0, 1, 2])

        metrics = compute_evaluation_metrics(y_true, y_pred)

        assert metrics["class_distribution"]["Safe"] == 3
        assert metrics["class_distribution"]["Stress"] == 1
        assert metrics["class_distribution"]["Danger"] == 1


class TestLoadClassifier:
    """Tests for classifier loading."""

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing model."""
        with pytest.raises(FileNotFoundError):
            load_classifier(tmp_path)


class TestLoadAndValidateData:
    """Tests for data loading and validation."""

    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        """Test that valid CSV is loaded correctly."""
        # Create test CSV
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "ph": [7.0] * 10,
            "tds": [300] * 10,
            "feature1": [1.0] * 10,
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        loaded_df, missing = load_and_validate_data(csv_path, ["ph", "tds"])

        assert len(loaded_df) == 10
        assert missing == []

    def test_reports_missing_features(self, tmp_path: Path) -> None:
        """Test that missing features are reported."""
        # Create test CSV without required features
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "ph": [7.0] * 10,
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        _, missing = load_and_validate_data(
            csv_path, ["ph", "tds", "ammonia_risk"])

        assert "tds" in missing
        assert "ammonia_risk" in missing

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_and_validate_data(tmp_path / "nonexistent.csv", ["ph"])
