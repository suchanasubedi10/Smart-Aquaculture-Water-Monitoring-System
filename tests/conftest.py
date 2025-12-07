"""
Pytest configuration and shared fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add source directories to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "backend"))


@pytest.fixture
def sample_reading() -> dict:
    """Sample sensor reading payload."""
    return {
        "timestamp": "2024-01-01T12:00:00Z",
        "ph": 7.4,
        "tds": 350,
        "device_id": "test_device",
    }


@pytest.fixture
def sample_readings_list() -> list[dict]:
    """List of sample readings for forecasting."""
    import pandas as pd

    base_time = pd.Timestamp("2024-01-01T12:00:00")
    readings = []
    for i in range(25):
        readings.append({
            "timestamp": (base_time + pd.Timedelta(minutes=5 * i)).isoformat(),
            "ph": 7.0 + 0.1 * (i % 5),
            "tds": 300 + 10 * (i % 5),
        })
    return readings


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    import numpy as np
    import pandas as pd

    n = 100
    rng = pd.date_range("2024-01-01", periods=n, freq="5min")

    return pd.DataFrame({
        "timestamp": rng,
        "ph": np.clip(7 + 0.5 * np.random.randn(n), 5.5, 9.5),
        "tds": np.clip(300 + 50 * np.random.randn(n), 50, 800),
        "temp": np.clip(25 + 3 * np.random.randn(n), 15, 35),
        "do": np.clip(7 + 1 * np.random.randn(n), 2, 12),
        "turbidity": np.clip(50 + 20 * np.random.randn(n), 0, 200),
    })
