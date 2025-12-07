"""
LSTM Utilities for Water Quality Forecasting.

Provides lazy-loaded LSTM models for pH and TDS time series forecasting
using PyTorch.

Commit message: "backend: harden LSTM utils, add lazy-load and validation"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch import nn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Path configuration
ROOT = Path(__file__).parent.resolve()
# Go up two levels from src/backend to reach project root
PROJECT_ROOT = ROOT.parent.parent

# Handle MODEL_DIR - if env var is relative, resolve from PROJECT_ROOT
_models_env = os.getenv("MODELS_DIR")
if _models_env:
    _models_path = Path(_models_env)
    if not _models_path.is_absolute():
        MODEL_DIR = (PROJECT_ROOT / _models_path).resolve()
    else:
        MODEL_DIR = _models_path.resolve()
else:
    MODEL_DIR = (PROJECT_ROOT / "models").resolve()

# Default sequence length (can be overridden by model card)
DEFAULT_SEQ_LEN = int(os.getenv("LSTM_SEQ_LEN", "20"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.

    Must match the architecture used during training.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units in LSTM.
            num_layers: Number of stacked LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y.squeeze(-1)


class LSTMModelRegistry:
    """Lazy-loading registry for LSTM models and scaler."""

    _instance: Optional["LSTMModelRegistry"] = None

    def __new__(cls) -> "LSTMModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ph_model = None
            cls._instance._tds_model = None
            cls._instance._scaler = None
            cls._instance._seq_len = DEFAULT_SEQ_LEN
            cls._instance._hidden_size = 64
            cls._instance._num_layers = 1
            cls._instance._loaded = False
            cls._instance._available = False
            cls._instance._model_cards = {}
        return cls._instance

    def _load_model_card(self, model_name: str) -> Optional[dict]:
        """Load model card JSON if it exists."""
        card_path = MODEL_DIR / f"{model_name}.model_card.json"
        if card_path.exists():
            try:
                with open(card_path) as f:
                    card = json.load(f)
                    logger.debug(f"Loaded model card: {card_path}")
                    return card
            except Exception as e:
                logger.warning(f"Failed to load model card {card_path}: {e}")
        return None

    def load(self) -> bool:
        """
        Load LSTM models and scaler.

        Returns:
            True if loading was successful, False otherwise.
        """
        if self._loaded:
            return self._available

        logger.info(f"Loading LSTM models from {MODEL_DIR}")

        try:
            # Load model cards first to get configuration
            ph_card = self._load_model_card("forecast_ph_lstm_pt")
            tds_card = self._load_model_card("forecast_tds_lstm_pt")

            if ph_card:
                self._model_cards["ph"] = ph_card
                # Get seq_len from model card if available
                if "seq_len" in ph_card:
                    card_seq_len = ph_card["seq_len"]
                    if card_seq_len != self._seq_len:
                        logger.warning(
                            f"Model card seq_len ({card_seq_len}) differs from default ({self._seq_len}). "
                            f"Using model card value."
                        )
                        self._seq_len = card_seq_len
                # Get hidden size and num_layers
                if "hidden_size" in ph_card:
                    self._hidden_size = ph_card["hidden_size"]
                if "num_layers" in ph_card:
                    self._num_layers = ph_card["num_layers"]

            if tds_card:
                self._model_cards["tds"] = tds_card

            # Load scaler
            scaler_path = MODEL_DIR / "lstm_scaler_pytorch.pkl"
            if not scaler_path.exists():
                logger.warning(f"Scaler not found: {scaler_path}")
                self._loaded = True
                return False

            self._scaler = joblib.load(scaler_path)
            logger.debug(f"Loaded scaler: {scaler_path}")

            # Load pH model
            ph_model_path = MODEL_DIR / "forecast_ph_lstm_pt.pth"
            if ph_model_path.exists():
                self._ph_model = LSTMModel(
                    input_size=2,
                    hidden_size=self._hidden_size,
                    num_layers=self._num_layers,
                ).to(DEVICE)
                self._ph_model.load_state_dict(
                    torch.load(ph_model_path, map_location=DEVICE)
                )
                self._ph_model.eval()
                logger.info(f"Loaded pH LSTM model: {ph_model_path}")
            else:
                logger.warning(f"pH model not found: {ph_model_path}")

            # Load TDS model
            tds_model_path = MODEL_DIR / "forecast_tds_lstm_pt.pth"
            if tds_model_path.exists():
                self._tds_model = LSTMModel(
                    input_size=2,
                    hidden_size=self._hidden_size,
                    num_layers=self._num_layers,
                ).to(DEVICE)
                self._tds_model.load_state_dict(
                    torch.load(tds_model_path, map_location=DEVICE)
                )
                self._tds_model.eval()
                logger.info(f"Loaded TDS LSTM model: {tds_model_path}")
            else:
                logger.warning(f"TDS model not found: {tds_model_path}")

            self._available = (
                self._ph_model is not None
                and self._tds_model is not None
                and self._scaler is not None
            )
            self._loaded = True

            if self._available:
                logger.info(
                    f"LSTM models ready (seq_len={self._seq_len}, "
                    f"hidden_size={self._hidden_size}, device={DEVICE})"
                )
            else:
                logger.warning("LSTM models not fully available")

            return self._available

        except Exception as e:
            logger.error(f"Failed to load LSTM models: {e}")
            self._loaded = True
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        """Check if LSTM models are available."""
        if not self._loaded:
            self.load()
        return self._available

    @property
    def ph_model(self) -> Optional[LSTMModel]:
        if not self._loaded:
            self.load()
        return self._ph_model

    @property
    def tds_model(self) -> Optional[LSTMModel]:
        if not self._loaded:
            self.load()
        return self._tds_model

    @property
    def scaler(self) -> Any:
        if not self._loaded:
            self.load()
        return self._scaler

    @property
    def seq_len(self) -> int:
        if not self._loaded:
            self.load()
        return self._seq_len

    @property
    def model_cards(self) -> dict[str, dict]:
        if not self._loaded:
            self.load()
        return self._model_cards


# Global registry instance
_registry = LSTMModelRegistry()


def get_lstm_registry() -> LSTMModelRegistry:
    """Get the global LSTM model registry."""
    return _registry


def validate_input_rows(last_rows: list[dict]) -> pd.DataFrame:
    """
    Validate and convert input rows to DataFrame.

    Args:
        last_rows: List of dictionaries with 'timestamp', 'ph', 'tds' keys.

    Returns:
        Validated and sorted DataFrame.

    Raises:
        ValueError: If input is invalid.
    """
    if not isinstance(last_rows, list):
        raise ValueError(f"Expected list, got {type(last_rows).__name__}")

    if len(last_rows) == 0:
        raise ValueError("Input list is empty")

    # Check required keys
    required_keys = {"timestamp", "ph", "tds"}
    for i, row in enumerate(last_rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not a dictionary")
        missing = required_keys - set(row.keys())
        if missing:
            raise ValueError(f"Row {i} missing required keys: {missing}")

    df = pd.DataFrame(last_rows)

    # Validate and convert types
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as e:
        raise ValueError(f"Invalid timestamp format: {e}")

    try:
        df["ph"] = pd.to_numeric(df["ph"], errors="raise")
        df["tds"] = pd.to_numeric(df["tds"], errors="raise")
    except Exception as e:
        raise ValueError(f"Invalid numeric values: {e}")

    # Validate ranges
    if (df["ph"] < 0).any() or (df["ph"] > 14).any():
        raise ValueError("pH values must be between 0 and 14")
    if (df["tds"] < 0).any():
        raise ValueError("TDS values must be non-negative")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def lstm_forecast(
    last_rows: list[dict],
    steps: int = 5,
) -> dict[str, list[float]]:
    """
    Forecast future pH and TDS values using LSTM models.

    Args:
        last_rows: List of recent readings as dictionaries with keys:
                  'timestamp', 'ph', 'tds'. Most recent should be last.
        steps: Number of future time steps to forecast.

    Returns:
        Dictionary with 'ph' and 'tds' lists containing predicted values
        in original scale.

    Example:
        >>> rows = [
        ...     {"timestamp": "2024-01-01T00:00:00", "ph": 7.0, "tds": 300},
        ...     {"timestamp": "2024-01-01T00:05:00", "ph": 7.1, "tds": 305},
        ...     # ... more rows
        ... ]
        >>> result = lstm_forecast(rows, steps=5)
        >>> print(result)
        {'ph': [7.05, 7.02, ...], 'tds': [302, 298, ...]}
    """
    registry = get_lstm_registry()

    if not registry.is_available:
        logger.warning("LSTM models not available, returning empty forecast")
        return {"ph": [], "tds": []}

    # Validate input
    try:
        df = validate_input_rows(last_rows)
    except ValueError as e:
        logger.error(f"Input validation failed: {e}")
        return {"ph": [], "tds": []}

    arr = df[["ph", "tds"]].values.astype(np.float32)
    seq_len = registry.seq_len
    scaler = registry.scaler

    # Pad if too short
    if len(arr) < seq_len:
        logger.debug(
            f"Input length ({len(arr)}) < seq_len ({seq_len}), padding with zeros")
        pad = np.zeros((seq_len - len(arr), 2), dtype=np.float32)
        arr = np.vstack([pad, arr])

    # Scale
    scaled = scaler.transform(arr)
    seq_scaled = scaled[-seq_len:].copy()

    ph_out: list[float] = []
    tds_out: list[float] = []

    ph_model = registry.ph_model
    tds_model = registry.tds_model

    for _ in range(steps):
        x = torch.from_numpy(seq_scaled.reshape(
            1, seq_len, 2)).float().to(DEVICE)

        with torch.no_grad():
            ph_pred_scaled = ph_model(x).cpu().numpy().item()
            tds_pred_scaled = tds_model(x).cpu().numpy().item()

        # Inverse transform
        ph_inv = scaler.inverse_transform([[ph_pred_scaled, 0.0]])[0][0]
        tds_inv = scaler.inverse_transform([[0.0, tds_pred_scaled]])[0][1]

        ph_out.append(float(ph_inv))
        tds_out.append(float(tds_inv))

        # Append predicted for next recursive input
        new_row_scaled = np.array(
            [ph_pred_scaled, tds_pred_scaled]).reshape(1, 2)
        seq_scaled = np.vstack([seq_scaled[1:], new_row_scaled])

    return {"ph": ph_out, "tds": tds_out}
