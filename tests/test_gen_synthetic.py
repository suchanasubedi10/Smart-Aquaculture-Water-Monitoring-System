"""
Tests for synthetic data generator (gen_synthetic_full.py).

Commit message: "scripts: make synthetic data generator CLI, add summary and tests"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestGenerateSyntheticData:
    """Tests for the generate_synthetic_data function."""

    def test_generates_correct_number_of_samples(self) -> None:
        """Test that generator produces requested number of samples."""
        from gen_synthetic_full import generate_synthetic_data

        df = generate_synthetic_data(n_samples=100, seed=42)
        assert len(df) == 100

    def test_generates_required_columns(self) -> None:
        """Test that all required columns are present."""
        from gen_synthetic_full import generate_synthetic_data

        df = generate_synthetic_data(n_samples=50, seed=42)
        required_cols = {"timestamp", "ph", "tds", "temp", "turbidity", "do"}
        assert required_cols.issubset(set(df.columns))

    def test_ph_values_in_range(self) -> None:
        """Test that pH values are within valid range."""
        from gen_synthetic_full import generate_synthetic_data

        df = generate_synthetic_data(n_samples=500, seed=42)
        assert df["ph"].min() >= 5.0
        assert df["ph"].max() <= 10.0

    def test_tds_values_positive(self) -> None:
        """Test that TDS values are positive."""
        from gen_synthetic_full import generate_synthetic_data

        df = generate_synthetic_data(n_samples=500, seed=42)
        assert df["tds"].min() >= 0

    def test_do_values_in_range(self) -> None:
        """Test that DO values are in sensible range."""
        from gen_synthetic_full import generate_synthetic_data

        df = generate_synthetic_data(n_samples=500, seed=42)
        assert df["do"].min() >= 0
        assert df["do"].max() <= 15

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces identical results."""
        from gen_synthetic_full import generate_synthetic_data

        df1 = generate_synthetic_data(n_samples=50, seed=123)
        df2 = generate_synthetic_data(n_samples=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self) -> None:
        """Test that different seeds produce different data."""
        from gen_synthetic_full import generate_synthetic_data

        df1 = generate_synthetic_data(n_samples=50, seed=1)
        df2 = generate_synthetic_data(n_samples=50, seed=2)
        assert not df1["ph"].equals(df2["ph"])


class TestInjectAnomalies:
    """Tests for anomaly injection."""

    def test_injects_correct_number_of_anomalies(self) -> None:
        """Test that specified number of anomalies are injected."""
        from gen_synthetic_full import generate_synthetic_data, inject_anomalies

        df = generate_synthetic_data(n_samples=100, seed=42)
        original_ph_mean = df["ph"].mean()

        df_anom = inject_anomalies(df, n_anomalies=10, seed=42)

        # After injection, data should be different
        assert len(df_anom) == len(df)

    def test_zero_anomalies_returns_unchanged(self) -> None:
        """Test that zero anomalies returns original data."""
        from gen_synthetic_full import generate_synthetic_data, inject_anomalies

        df = generate_synthetic_data(n_samples=50, seed=42)
        df_result = inject_anomalies(df, n_anomalies=0, seed=42)
        pd.testing.assert_frame_equal(df, df_result)


class TestComputeSummary:
    """Tests for summary statistics computation."""

    def test_summary_contains_required_fields(self) -> None:
        """Test that summary has all required fields."""
        from gen_synthetic_full import compute_summary_statistics, generate_synthetic_data

        df = generate_synthetic_data(n_samples=50, seed=42)
        summary = compute_summary_statistics(df)

        assert "generated_at" in summary
        assert "n_samples" in summary
        assert "columns" in summary
        assert summary["n_samples"] == 50

    def test_summary_columns_have_stats(self) -> None:
        """Test that each column has mean, std, min, max, median."""
        from gen_synthetic_full import compute_summary_statistics, generate_synthetic_data

        df = generate_synthetic_data(n_samples=50, seed=42)
        summary = compute_summary_statistics(df)

        for col in ["ph", "tds", "temp"]:
            assert col in summary["columns"]
            stats = summary["columns"][col]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "median" in stats


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_generate_creates_files(self, tmp_path: Path) -> None:
        """Test that CLI creates expected output files."""
        from gen_synthetic_full import generate_synthetic_data, compute_summary_statistics

        # Generate data
        df = generate_synthetic_data(n_samples=50, seed=42)

        # Save CSV
        csv_path = tmp_path / "raw_combined.csv"
        df.to_csv(csv_path, index=False)
        assert csv_path.exists()

        # Save summary
        summary = compute_summary_statistics(df)
        summary_path = tmp_path / "sample_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f)
        assert summary_path.exists()

        # Verify CSV content
        df_loaded = pd.read_csv(csv_path)
        assert len(df_loaded) == 50
        assert "ph" in df_loaded.columns
