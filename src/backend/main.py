"""
Water Intelligence API - FastAPI Backend.

Provides REST API endpoints for water quality prediction, forecasting,
and recommendations.

Commit message: "backend: add pydantic models, CORS, startup model load, /api/models endpoint"
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from src.backend.lstm_utils_pytorch import get_lstm_registry, lstm_forecast
from src.backend.model_utils import (
    Recommendation,
    classify,
    compute_basic_features,
    detect_anomaly,
    get_model_registry,
    predict_forecast_rf,
    predict_virtual_sensors,
    recommend,
)
from src.backend.routes.generative import router as generative_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
CORS_ORIGINS = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
).split(",")
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))


# ========================
# Pydantic Models
# ========================


class Reading(BaseModel):
    """Single sensor reading input."""

    timestamp: str = Field(..., description="ISO 8601 timestamp string")
    ph: float = Field(..., ge=0, le=14, description="pH value (0-14)")
    tds: float = Field(..., ge=0, description="Total Dissolved Solids (mg/L)")
    device_id: str = Field(default="device_1", description="Device identifier")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-01T12:00:00Z",
                "ph": 7.4,
                "tds": 350,
                "device_id": "sensor_1",
            }
        }


class ForecastRequest(BaseModel):
    """Request for time series forecast."""

    last_rows: list[dict] = Field(
        ..., description="List of recent readings with timestamp, ph, tds"
    )
    steps: int = Field(default=5, ge=1, le=100,
                       description="Number of future steps")

    @validator("last_rows")
    def validate_last_rows(cls, v: list[dict]) -> list[dict]:
        if len(v) == 0:
            raise ValueError("last_rows cannot be empty")
        required = {"timestamp", "ph", "tds"}
        for i, row in enumerate(v):
            missing = required - set(row.keys())
            if missing:
                raise ValueError(f"Row {i} missing: {missing}")
        return v


class RecommendationResponse(BaseModel):
    """Structured recommendation."""

    code: str
    severity: str
    message: str


class VirtualSensorsResponse(BaseModel):
    """Virtual sensor predictions."""

    temp_est: Optional[float] = None
    do_est: Optional[float] = None
    turbidity_est: Optional[float] = None


class AnomalyResponse(BaseModel):
    """Anomaly detection result."""

    is_anomaly: bool
    score: Optional[float] = None


class ClassificationResponse(BaseModel):
    """Classification result."""

    prediction: Optional[int] = Field(
        None, description="0=Safe, 1=Stress, 2=Danger")
    probabilities: Optional[list[float]] = None


class PredictResponse(BaseModel):
    """Complete prediction response."""

    features: dict[str, Any]
    virtual: VirtualSensorsResponse
    anomaly: AnomalyResponse
    classification: ClassificationResponse
    recommendations: list[str]
    recommendations_structured: list[RecommendationResponse] = []


class ForecastResponse(BaseModel):
    """Forecast response."""

    ph_forecast: list[float] = []
    tds_forecast: list[float] = []


class LSTMForecastResponse(BaseModel):
    """LSTM forecast response."""

    ph: list[float] = []
    tds: list[float] = []


class ModelInfo(BaseModel):
    """Model metadata."""

    name: str
    available: bool
    version: Optional[str] = None
    created_at: Optional[str] = None
    model_type: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response for /api/models endpoint."""

    models: list[ModelInfo]
    lstm_available: bool
    model_dir: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "1.0.0"


# ========================
# Simple Rate Limiter (demo only)
# ========================


class SimpleRateLimiter:
    """Simple in-memory rate limiter for demonstration."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.rpm = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make a request."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > minute_ago
        ]

        if len(self.requests[client_id]) >= self.rpm:
            return False

        self.requests[client_id].append(now)
        return True


rate_limiter = SimpleRateLimiter(RATE_LIMIT_RPM)


# ========================
# Lifespan / Startup
# ========================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: warm-load models
    logger.info("=" * 60)
    logger.info("Starting Water Intelligence API")
    logger.info("=" * 60)

    # Load classical models
    registry = get_model_registry()
    registry.load_all()

    # Load LSTM models
    lstm_registry = get_lstm_registry()
    lstm_registry.load()

    # Log model status
    model_cards = registry.get_all_cards()
    for name, card in model_cards.items():
        logger.info(
            f"Model: {name} | Version: {card.get('version', 'N/A')[:8]}... | "
            f"Created: {card.get('created_at', 'N/A')}"
        )

    if lstm_registry.is_available:
        logger.info(f"LSTM models ready (seq_len={lstm_registry.seq_len})")
    else:
        logger.warning("LSTM models not available")

    logger.info("API ready to serve requests")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down API")


# ========================
# FastAPI App
# ========================

app = FastAPI(
    title="Water Intelligence API",
    description="API for water quality prediction, forecasting, and recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Include routers
app.include_router(generative_router)


# ========================
# Exception Handlers
# ========================


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueError exceptions."""
    logger.warning(f"ValueError: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": "Bad Request", "detail": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """Handle RuntimeError exceptions."""
    logger.error(f"RuntimeError: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# ========================
# Dependency Injection
# ========================


def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    """Check rate limit for client."""
    if not RATE_LIMIT_ENABLED:
        return

    client_id = get_client_id(request)
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )


# ========================
# API Endpoints
# ========================


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/api/models", response_model=ModelsResponse, tags=["Models"])
def list_models() -> ModelsResponse:
    """
    List all available models and their metadata.

    Returns model names, versions, and availability status.
    """
    registry = get_model_registry()
    lstm_registry = get_lstm_registry()

    model_names = [
        "virtual_temp",
        "virtual_do",
        "virtual_turb",
        "forecast_ph_rf",
        "forecast_tds_rf",
        "anomaly",
        "classifier",
    ]

    models: list[ModelInfo] = []
    model_cards = registry.get_all_cards()

    for name in model_names:
        available = registry.is_available(name)
        card = model_cards.get(name, {})
        models.append(
            ModelInfo(
                name=name,
                available=available,
                version=card.get("version"),
                created_at=card.get("created_at"),
                model_type=card.get("model_type"),
            )
        )

    # Add LSTM models
    for lstm_name in ["forecast_ph_lstm", "forecast_tds_lstm"]:
        card = lstm_registry.model_cards.get(lstm_name.split("_")[-1], {})
        models.append(
            ModelInfo(
                name=lstm_name,
                available=lstm_registry.is_available,
                version=card.get("version"),
                created_at=card.get("created_at"),
                model_type="LSTMModel",
            )
        )

    return ModelsResponse(
        models=models,
        lstm_available=lstm_registry.is_available,
        model_dir=str(registry._loaded and "loaded" or "not loaded"),
    )


@app.post("/api/feature_row", tags=["Features"])
def feature_row(
    r: Reading,
    _: None = Depends(check_rate_limit),
) -> dict[str, Any]:
    """
    Compute features from a single sensor reading.

    Returns the computed feature vector used by prediction models.
    """
    row = compute_basic_features(r.dict())
    return row.to_dict(orient="records")[0]


@app.post("/api/virtual", tags=["Prediction"])
def virtual(
    r: Reading,
    _: None = Depends(check_rate_limit),
) -> dict[str, Any]:
    """
    Predict virtual sensor values (temperature, DO, turbidity).

    Uses trained regression models to estimate sensor values
    from pH and TDS inputs.
    """
    features_df = compute_basic_features(r.dict())
    virt = predict_virtual_sensors(features_df)
    return {"virtual": virt, "features": features_df.to_dict(orient="records")[0]}


@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
def endpoint_predict(
    r: Reading,
    _: None = Depends(check_rate_limit),
) -> PredictResponse:
    """
    Run complete prediction pipeline.

    Includes:
    - Feature computation
    - Virtual sensor prediction
    - Anomaly detection
    - Classification (Safe/Stress/Danger)
    - Recommendations
    """
    # Compute features
    features_df = compute_basic_features(r.dict())
    features_dict = features_df.to_dict(orient="records")[0]

    # Virtual sensors
    virt = predict_virtual_sensors(features_df)

    # Anomaly detection
    anom = detect_anomaly(features_df)

    # Classification
    cls = classify(features_df)

    # Recommendations
    all_out = {
        "features": features_dict,
        "virtual": virt,
        "anomaly": anom,
        "classification": cls,
    }
    recs = recommend(all_out)

    return PredictResponse(
        features=features_dict,
        virtual=VirtualSensorsResponse(**virt),
        anomaly=AnomalyResponse(**anom),
        classification=ClassificationResponse(**cls),
        recommendations=[rec.message for rec in recs],
        recommendations_structured=[
            RecommendationResponse(
                code=rec.code,
                severity=rec.severity,
                message=rec.message,
            )
            for rec in recs
        ],
    )


@app.post("/api/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def endpoint_forecast(
    last_rows: list[dict],
    steps: int = 3,
    _: None = Depends(check_rate_limit),
) -> ForecastResponse:
    """
    Forecast future pH and TDS using Random Forest models.

    Args:
        last_rows: List of recent readings with timestamp, ph, tds.
        steps: Number of future time steps to predict.

    Returns:
        Predicted pH and TDS values for each future step.
    """
    if not isinstance(last_rows, list) or len(last_rows) == 0:
        raise HTTPException(
            status_code=400,
            detail="Provide last_rows list with timestamp, ph, tds",
        )

    df = pd.DataFrame(last_rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    res = predict_forecast_rf(df, steps=steps)
    return ForecastResponse(**res)


@app.post("/api/forecast_lstm", response_model=LSTMForecastResponse, tags=["Forecasting"])
def endpoint_forecast_lstm(
    last_rows: list[dict],
    steps: int = 5,
    _: None = Depends(check_rate_limit),
) -> LSTMForecastResponse:
    """
    Forecast future pH and TDS using LSTM models.

    Uses PyTorch LSTM models for time series forecasting.
    Requires at least seq_len (default 20) recent readings for best results.

    Args:
        last_rows: List of recent readings with timestamp, ph, tds.
        steps: Number of future time steps to predict.

    Returns:
        Predicted pH and TDS values for each future step.
    """
    if not isinstance(last_rows, list) or len(last_rows) == 0:
        raise HTTPException(
            status_code=400,
            detail="Provide last_rows list with timestamp, ph, tds",
        )

    result = lstm_forecast(last_rows, steps=steps)
    return LSTMForecastResponse(**result)


# ========================
# Main Entry Point
# ========================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
