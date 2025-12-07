"""
Model Utilities for Water Intelligence Backend.

Provides functions for feature computation, model inference, anomaly detection,
classification, and recommendation generation.

Commit message: "backend: improve model utils, add types, lazy loading, structured recommendations"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, validator

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


# ========================
# Pydantic Input Validation
# ========================


class SensorReading(BaseModel):
    """Validated sensor reading input."""

    timestamp: str
    ph: float
    tds: float
    device_id: str = "unknown"

    @validator("ph")
    def validate_ph(cls, v: float) -> float:
        if not 0 <= v <= 14:
            raise ValueError(f"pH must be between 0 and 14, got {v}")
        return v

    @validator("tds")
    def validate_tds(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"TDS must be non-negative, got {v}")
        return v


# ========================
# Result Dataclasses
# ========================


@dataclass
class Recommendation:
    """Structured recommendation."""

    code: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str


@dataclass
class PredictionResult:
    """Complete prediction result container."""

    features: dict[str, Any]
    virtual: dict[str, Optional[float]]
    anomaly: dict[str, Any]
    classification: dict[str, Any]
    recommendations: list[Recommendation]
    recommendations_text: list[str]  # Legacy text format for compatibility
    metadata: dict[str, Any] = field(default_factory=dict)


# ========================
# Lazy Model Loading
# ========================


class ModelRegistry:
    """Lazy-loading model registry with caching."""

    _instance: Optional["ModelRegistry"] = None
    _models: dict[str, Any]
    _model_cards: dict[str, dict]
    _scaler: Any
    _feature_list: list[str]
    _loaded: bool

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._model_cards = {}
            cls._instance._scaler = None
            cls._instance._feature_list = []
            cls._instance._loaded = False
        return cls._instance

    def _safe_load(self, name: str) -> Any:
        """Safely load a joblib model file."""
        path = MODEL_DIR / name
        if path.exists():
            try:
                model = joblib.load(path)
                logger.debug(f"Loaded model: {name}")
                return model
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        return None

    def _load_model_card(self, model_name: str) -> Optional[dict]:
        """Load model card JSON if it exists."""
        card_path = MODEL_DIR / f"{model_name}.model_card.json"
        if card_path.exists():
            try:
                with open(card_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model card {card_path}: {e}")
        return None

    def load_all(self) -> None:
        """Load all models and log availability."""
        if self._loaded:
            return

        logger.info(f"Loading models from {MODEL_DIR}")

        model_files = {
            "virtual_temp": "virtual_temp_reg.pkl",
            "virtual_do": "virtual_do_reg.pkl",
            "virtual_turb": "virtual_turb_reg.pkl",
            "forecast_ph_rf": "forecast_ph_rf.pkl",
            "forecast_tds_rf": "forecast_tds_rf.pkl",
            "anomaly": "anomaly_if.pkl",
            "classifier": "classifier_rf.pkl",
        }

        for key, filename in model_files.items():
            self._models[key] = self._safe_load(filename)
            card = self._load_model_card(filename.replace(".pkl", ""))
            if card:
                self._model_cards[key] = card

        # Load scaler
        self._scaler = self._safe_load("feature_scaler.pkl")

        # Load feature list
        fl_path = MODEL_DIR / "feature_list.json"
        if fl_path.exists():
            try:
                with open(fl_path) as f:
                    self._feature_list = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load feature list: {e}")

        # Log availability
        available = [k for k, v in self._models.items() if v is not None]
        logger.info(f"Available models: {available}")
        self._loaded = True

    def get(self, name: str) -> Any:
        """Get a loaded model by name."""
        if not self._loaded:
            self.load_all()
        return self._models.get(name)

    def get_all_cards(self) -> dict[str, dict]:
        """Get all model cards."""
        if not self._loaded:
            self.load_all()
        return self._model_cards

    @property
    def scaler(self) -> Any:
        if not self._loaded:
            self.load_all()
        return self._scaler

    @property
    def feature_list(self) -> list[str]:
        if not self._loaded:
            self.load_all()
        return self._feature_list

    def is_available(self, name: str) -> bool:
        """Check if a model is available."""
        return self.get(name) is not None


# Global registry instance
_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry


# ========================
# Feature Computation
# ========================


def compute_basic_features(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Compute features from a sensor reading payload.

    Args:
        payload: Dictionary with keys 'ph', 'tds', 'timestamp' (ISO string).

    Returns:
        DataFrame with single row containing features used by models.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    try:
        reading = SensorReading(**payload)
    except ValidationError as e:
        raise ValueError(f"Invalid sensor reading: {e}")

    ph = reading.ph
    tds = reading.tds
    ts = pd.to_datetime(reading.timestamp)
    hour = int(ts.hour)

    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    is_dawn = 1 if 4 <= hour <= 6 else 0

    # Rolling placeholders for single-row input
    ph_roll = ph
    tds_roll = tds
    delta_ph = 0.0
    delta_tds = 0.0

    # Ammonia risk calculation
    registry = get_model_registry()
    scaler = registry.scaler

    temp_est = 20 + 10 * (tds - 50) / max(1, (1500 - 50))
    ammonia_raw = ph * temp_est

    if scaler is not None:
        try:
            ammonia_norm = float(scaler.transform([[ammonia_raw]])[0][0])
        except Exception:
            ammonia_norm = float(ammonia_raw / 500.0)
    else:
        ammonia_norm = float(ammonia_raw / 500.0)

    row = {
        "ph": ph,
        "tds": tds,
        "ammonia_risk": ammonia_norm,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "is_dawn": is_dawn,
        "ph_roll": ph_roll,
        "tds_roll": tds_roll,
        "delta_ph": delta_ph,
        "delta_tds": delta_tds,
    }

    df = pd.DataFrame([row])

    # Reorder to match training feature order
    feature_list = registry.feature_list
    if feature_list:
        for f in feature_list:
            if f not in df.columns:
                if f == "ph_roll_mean" and "ph_roll" in df.columns:
                    df[f] = df["ph_roll"]
                elif f == "tds_roll_mean" and "tds_roll" in df.columns:
                    df[f] = df["tds_roll"]
                else:
                    df[f] = 0.0
        df = df[feature_list]

    return df


# ========================
# Virtual Sensor Predictions
# ========================


def predict_virtual_sensors(features_df: pd.DataFrame) -> dict[str, Optional[float]]:
    """
    Predict virtual sensor values (temp, DO, turbidity).

    Args:
        features_df: DataFrame with features for virtual regressors.

    Returns:
        Dictionary with estimated virtual sensor values.
    """
    registry = get_model_registry()
    out: dict[str, Optional[float]] = {}

    sensor_models = [
        ("virtual_temp", "temp_est"),
        ("virtual_do", "do_est"),
        ("virtual_turb", "turbidity_est"),
    ]

    for model_key, output_key in sensor_models:
        model = registry.get(model_key)
        if model is not None:
            try:
                # Get the required features for virtual sensors
                virt_features = ["ph", "tds", "ph_roll",
                                 "tds_roll", "hour_sin", "hour_cos", "is_dawn"]
                available_features = [
                    f for f in virt_features if f in features_df.columns]

                if len(available_features) == len(virt_features):
                    input_df = features_df[virt_features]
                else:
                    # Fallback: use available columns
                    input_df = features_df

                out[output_key] = float(model.predict(input_df)[0])
            except Exception as e:
                logger.warning(
                    f"Virtual sensor prediction failed for {model_key}: {e}")
                out[output_key] = None
        else:
            out[output_key] = None

    return out


# ========================
# RF Forecast
# ========================


def predict_forecast_rf(
    last_rows_df: pd.DataFrame,
    steps: int = 3,
) -> dict[str, list[float]]:
    """
    Predict future pH and TDS values using Random Forest models.

    Args:
        last_rows_df: DataFrame with columns ['timestamp', 'ph', 'tds'].
                     Should contain at least 5 recent readings.
        steps: Number of future steps to predict.

    Returns:
        Dictionary with 'ph_forecast' and 'tds_forecast' lists.
    """
    res: dict[str, list[float]] = {"ph_forecast": [], "tds_forecast": []}

    registry = get_model_registry()
    rf_ph = registry.get("forecast_ph_rf")
    rf_tds = registry.get("forecast_tds_rf")

    if rf_ph is None or rf_tds is None:
        logger.warning("RF forecast models not available")
        return res

    df = last_rows_df.copy()

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        logger.warning("No timestamp column in input DataFrame")
        return res

    # Infer time delta safely
    if len(df) > 1:
        time_diffs = df["timestamp"].diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            if pd.isna(median_diff) or median_diff.total_seconds() <= 0:
                time_delta = pd.Timedelta(minutes=5)
            else:
                time_delta = median_diff
        else:
            time_delta = pd.Timedelta(minutes=5)
    else:
        time_delta = pd.Timedelta(minutes=5)

    logger.debug(f"Using time delta: {time_delta}")

    for _ in range(steps):
        # Get last 5 lags (or pad)
        if len(df) >= 5:
            ph_lags = df["ph"].iloc[-5:].values
            tds_lags = df["tds"].iloc[-5:].values
        else:
            ph_lags = np.concatenate([np.zeros(5 - len(df)), df["ph"].values])
            tds_lags = np.concatenate(
                [np.zeros(5 - len(df)), df["tds"].values])

        # Feature order expected by RF forecaster: ph_lag_1..5, tds_lag_1..5, hour_sin, hour_cos, is_dawn
        feat = np.concatenate([ph_lags[::-1], tds_lags[::-1]])

        # Time features for next step
        last_time = df["timestamp"].iloc[-1]
        next_time = last_time + time_delta
        hour = next_time.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        is_dawn = 1 if 4 <= hour <= 6 else 0

        feat = np.concatenate(
            [feat, [hour_sin, hour_cos, is_dawn]]).reshape(1, -1)

        ph_pred = float(rf_ph.predict(feat)[0])
        tds_pred = float(rf_tds.predict(feat)[0])

        res["ph_forecast"].append(ph_pred)
        res["tds_forecast"].append(tds_pred)

        # Append predicted for recursive forecasting
        new_row = pd.DataFrame(
            [{"timestamp": next_time, "ph": ph_pred, "tds": tds_pred}])
        df = pd.concat([df, new_row], ignore_index=True)

    return res


# ========================
# Anomaly Detection
# ========================


def detect_anomaly(features_df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect anomalies using IsolationForest model.

    Args:
        features_df: DataFrame with anomaly detection features.

    Returns:
        Dictionary with 'is_anomaly' boolean and 'score' float.
    """
    registry = get_model_registry()
    model = registry.get("anomaly")

    if model is None:
        return {"is_anomaly": False, "score": None}

    try:
        # Anomaly features
        anom_features = ["ph", "tds", "ammonia_risk",
                         "ph_roll", "tds_roll", "delta_ph", "delta_tds"]
        available = [f for f in anom_features if f in features_df.columns]

        if len(available) < len(anom_features):
            logger.warning(
                f"Missing anomaly features: {set(anom_features) - set(available)}")
            return {"is_anomaly": False, "score": None}

        input_df = features_df[anom_features]
        score = float(model.decision_function(input_df)[0])
        is_anom = int(model.predict(input_df)[0]) == -1
        return {"is_anomaly": bool(is_anom), "score": score}
    except Exception as e:
        logger.warning(f"Anomaly detection failed: {e}")
        return {"is_anomaly": False, "score": None}


# ========================
# Classification
# ========================


def classify(features_df: pd.DataFrame) -> dict[str, Any]:
    """
    Classify water quality (Safe=0, Stress=1, Danger=2).

    Args:
        features_df: DataFrame with classification features.

    Returns:
        Dictionary with 'prediction' and 'probabilities'.
    """
    registry = get_model_registry()
    model = registry.get("classifier")

    if model is None:
        return {"prediction": None, "probabilities": None}

    try:
        proba = model.predict_proba(features_df)[0].tolist()
        pred = int(model.predict(features_df)[0])
        return {"prediction": pred, "probabilities": proba}
    except Exception as e:
        logger.warning(f"Classification failed: {e}")
        return {"prediction": None, "probabilities": None}


# ========================
# Recommendations
# ========================


def recommend(all_outputs: dict[str, Any]) -> list[Recommendation]:
    """
    Generate actionable recommendations based on predictions.

    Args:
        all_outputs: Dictionary containing classification, virtual, anomaly, features.

    Returns:
        List of structured Recommendation objects.
    """
    recs: list[Recommendation] = []

    cls = all_outputs.get("classification", {})
    virt = all_outputs.get("virtual", {})
    anom = all_outputs.get("anomaly", {})
    features = all_outputs.get("features", {})

    pred = cls.get("prediction")
    do_est = virt.get("do_est")
    temp_est = virt.get("temp_est")
    turb_est = virt.get("turbidity_est")
    is_anom = anom.get("is_anomaly", False)
    ammonia_index = features.get("ammonia_risk")

    # Anomaly check
    if is_anom:
        recs.append(
            Recommendation(
                code="ANOMALY_DETECTED",
                severity="critical",
                message="Anomaly detected — check sensors and water source immediately.",
            )
        )

    # Classification-based recommendations
    if pred == 2:
        recs.append(
            Recommendation(
                code="DANGER_CONDITION",
                severity="critical",
                message="Water condition classified as DANGER. Immediate action: check aeration, reduce feeding, consider partial water exchange.",
            )
        )
    elif pred == 1:
        recs.append(
            Recommendation(
                code="STRESS_CONDITION",
                severity="high",
                message="Water condition classified as STRESS. Monitor closely and consider corrective steps.",
            )
        )
    else:
        recs.append(
            Recommendation(
                code="SAFE_CONDITION",
                severity="low",
                message="Water condition classified as SAFE.",
            )
        )

    # DO-based recommendations
    if do_est is not None:
        try:
            do_val = float(do_est)
            if do_val < 3.5:
                recs.append(
                    Recommendation(
                        code="DO_CRITICAL",
                        severity="critical",
                        message=f"Estimated DO is critically low ({do_val:.2f} mg/L). Increase aeration or water exchange immediately.",
                    )
                )
            elif do_val < 5.0:
                recs.append(
                    Recommendation(
                        code="DO_LOW",
                        severity="high",
                        message=f"Estimated DO is marginal ({do_val:.2f} mg/L). Monitor closely.",
                    )
                )
        except (ValueError, TypeError):
            pass

    # Ammonia-based recommendations
    if ammonia_index is not None:
        try:
            if float(ammonia_index) > 0.6:
                recs.append(
                    Recommendation(
                        code="AMMONIA_HIGH",
                        severity="high",
                        message="High ammonia risk detected. Consider partial water replacement and check feeding/organic load.",
                    )
                )
        except (ValueError, TypeError):
            pass

    # Turbidity-based recommendations
    if turb_est is not None:
        try:
            if float(turb_est) > 150:
                recs.append(
                    Recommendation(
                        code="TURBIDITY_HIGH",
                        severity="medium",
                        message="High turbidity predicted — check solids and reduce feed/waste.",
                    )
                )
        except (ValueError, TypeError):
            pass

    return recs


def recommend_legacy(all_outputs: dict[str, Any]) -> list[str]:
    """
    Generate recommendations as simple text strings (legacy format).

    Args:
        all_outputs: Dictionary containing prediction outputs.

    Returns:
        List of recommendation strings.
    """
    recs = recommend(all_outputs)
    return [r.message for r in recs]


def get_full_prediction(payload: dict[str, Any]) -> PredictionResult:
    """
    Run the complete prediction pipeline.

    Args:
        payload: Sensor reading payload.

    Returns:
        Complete PredictionResult with all predictions and recommendations.
    """
    features_df = compute_basic_features(payload)
    virtual = predict_virtual_sensors(features_df)
    anomaly = detect_anomaly(features_df)
    classification = classify(features_df)

    all_outputs = {
        "features": features_df.to_dict(orient="records")[0],
        "virtual": virtual,
        "anomaly": anomaly,
        "classification": classification,
    }

    recommendations = recommend(all_outputs)
    recommendations_text = [r.message for r in recommendations]

    return PredictionResult(
        features=all_outputs["features"],
        virtual=virtual,
        anomaly=anomaly,
        classification=classification,
        recommendations=recommendations,
        recommendations_text=recommendations_text,
        metadata={
            "model_dir": str(MODEL_DIR),
            "models_available": [k for k, v in _registry._models.items() if v is not None] if _registry._loaded else [],
        },
    )
