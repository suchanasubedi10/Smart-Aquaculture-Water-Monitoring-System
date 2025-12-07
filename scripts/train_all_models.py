#!/usr/bin/env python3
"""
Training Pipeline for Water Quality Models.

Trains classical ML models including:
- Virtual sensor regressors (temp, DO, turbidity)
- RF forecasters (pH, TDS)
- Anomaly detector (IsolationForest)
- Multi-class classifier (Safe/Stress/Danger)

Commit message: "scripts: refactor training pipeline, add metrics and model cards"
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Train all classical ML models for water quality prediction.")

# Default paths from environment
DEFAULT_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DEFAULT_MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
DEFAULT_SEED = int(os.getenv("RANDOM_SEED", "42"))
DEFAULT_TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

# Feature configurations
VIRT_FEATURES = ["ph", "tds", "ph_roll",
                 "tds_roll", "hour_sin", "hour_cos", "is_dawn"]
CLF_FEATURES = [
    "ph",
    "tds",
    "ammonia_risk",
    "hour_sin",
    "hour_cos",
    "is_dawn",
    "ph_roll",
    "tds_roll",
    "delta_ph",
    "delta_tds",
]
ANOM_FEATURES = [
    "ph",
    "tds",
    "ammonia_risk",
    "ph_roll",
    "tds_roll",
    "delta_ph",
    "delta_tds",
]


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.debug(f"Random seeds set to {seed}")


def load_data(path: Path) -> pd.DataFrame:
    """
    Load raw data from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with parsed timestamp column.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required_cols = {"timestamp", "ph", "tds"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {len(df)} rows")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by removing invalid values and IQR outliers.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame with outliers removed.
    """
    logger.info("Cleaning data...")

    # Basic validation
    df = df[(df["ph"] > 0) & (df["tds"] >= 0)].copy()

    def remove_iqr(col: pd.Series) -> pd.Series:
        """Remove outliers using IQR method."""
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return col.where((col >= lower) & (col <= upper))

    for c in ["ph", "tds", "temp", "turbidity", "do"]:
        if c in df.columns:
            df[c] = remove_iqr(df[c])

    df = df.dropna().reset_index(drop=True)
    logger.info(f"After cleaning: {len(df)} rows")
    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from the cleaned data.

    Args:
        df: Cleaned DataFrame with timestamp, ph, tds columns.

    Returns:
        DataFrame with engineered features.
    """
    logger.info("Engineering features...")
    df = df.copy()

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_dawn"] = ((df["hour"] >= 4) & (df["hour"] <= 6)).astype(int)

    # Rolling features
    df["ph_roll"] = df["ph"].rolling(window=5, min_periods=1).mean()
    df["tds_roll"] = df["tds"].rolling(window=5, min_periods=1).mean()

    # Delta features
    df["delta_ph"] = df["ph"] - df["ph"].shift(1).fillna(df["ph"].iloc[0])
    df["delta_tds"] = df["tds"] - df["tds"].shift(1).fillna(df["tds"].iloc[0])

    # Ammonia risk raw
    if "temp" in df.columns:
        df["ammonia_risk_raw"] = df["ph"] * df["temp"]
    else:
        tds_min, tds_max = df["tds"].min(), df["tds"].max()
        df["temp_proxy"] = 20 + 10 * \
            (df["tds"] - tds_min) / max(1, (tds_max - tds_min))
        df["ammonia_risk_raw"] = df["ph"] * df["temp_proxy"]

    logger.info(f"Featurized DataFrame has {len(df.columns)} columns")
    return df


def create_model_card(
    name: str,
    model_type: str,
    args: dict,
    feature_list: list[str],
    metrics: Optional[dict] = None,
    seq_len: Optional[int] = None,
) -> dict:
    """
    Create a model card with metadata.

    Args:
        name: Model name.
        model_type: Type of model (e.g., 'RandomForestRegressor').
        args: Training arguments used.
        feature_list: List of feature names.
        metrics: Optional metrics dictionary.
        seq_len: Sequence length for LSTM models.

    Returns:
        Dictionary containing model metadata.
    """
    card = {
        "name": name,
        "version": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "model_type": model_type,
        "training_args": args,
        "feature_list": feature_list,
        "metrics_summary": metrics or {},
    }
    if seq_len is not None:
        card["seq_len"] = seq_len
    return card


def save_model_artifact(
    model: Any,
    path: Path,
    model_card: dict,
) -> bool:
    """
    Save a model artifact with its model card.

    Args:
        model: The trained model object.
        path: Path to save the model.
        model_card: Model metadata dictionary.

    Returns:
        True if successful, False otherwise.
    """
    try:
        joblib.dump(model, path)
        logger.info(f"✔ Saved model: {path}")

        # Save model card
        card_path = path.with_suffix(".model_card.json")
        with open(card_path, "w") as f:
            json.dump(model_card, f, indent=2)
        logger.info(f"✔ Saved model card: {card_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to save model {path}: {e}")
        return False


def train_virtual_sensors(
    df: pd.DataFrame,
    models_dir: Path,
    test_size: float,
    seed: int,
) -> dict:
    """
    Train virtual sensor regressors for temp, DO, turbidity.

    Args:
        df: Featurized DataFrame.
        models_dir: Directory to save models.
        test_size: Fraction for test split.
        seed: Random seed.

    Returns:
        Dictionary with training metrics.
    """
    logger.info("Training virtual sensor regressors...")
    metrics: dict = {}
    training_args = {"n_estimators": 200, "max_depth": 12,
                     "test_size": test_size, "seed": seed}

    targets = [
        ("temp", "virtual_temp_reg.pkl"),
        ("do", "virtual_do_reg.pkl"),
        ("turbidity", "virtual_turb_reg.pkl"),
    ]

    for target_col, filename in targets:
        if target_col not in df.columns:
            logger.warning(f"Skipping {target_col} - column not found")
            continue

        X = df[VIRT_FEATURES]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        model = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=seed
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics[target_col] = {"rmse": rmse,
                               "n_train": len(X_train), "n_test": len(X_test)}
        logger.info(f"  {target_col} RMSE: {rmse:.4f}")

        # Save
        model_card = create_model_card(
            name=filename.replace(".pkl", ""),
            model_type="RandomForestRegressor",
            args=training_args,
            feature_list=VIRT_FEATURES,
            metrics=metrics[target_col],
        )
        save_model_artifact(model, models_dir / filename, model_card)

    return metrics


def train_forecasters(
    df: pd.DataFrame,
    models_dir: Path,
    test_size: float,
    seed: int,
) -> dict:
    """
    Train RF forecasters for pH and TDS.

    Args:
        df: Featurized DataFrame.
        models_dir: Directory to save models.
        test_size: Fraction for test split.
        seed: Random seed.

    Returns:
        Dictionary with training metrics.
    """
    logger.info("Training RF forecasters (ph & tds 1-step ahead)...")
    metrics: dict = {}
    training_args = {"n_estimators": 200, "max_depth": 12,
                     "test_size": test_size, "seed": seed}

    df_fore = df.copy()

    # Create lag features
    for lag in range(1, 6):
        df_fore[f"ph_lag_{lag}"] = df_fore["ph"].shift(lag).bfill()
        df_fore[f"tds_lag_{lag}"] = df_fore["tds"].shift(lag).bfill()

    forecast_features = (
        [f"ph_lag_{i}" for i in range(1, 6)]
        + [f"tds_lag_{i}" for i in range(1, 6)]
        + ["hour_sin", "hour_cos", "is_dawn"]
    )

    X = df_fore[forecast_features]

    for target_col, filename in [("ph", "forecast_ph_rf.pkl"), ("tds", "forecast_tds_rf.pkl")]:
        y = df_fore[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        model = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=seed
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics[target_col] = {"rmse": rmse,
                               "n_train": len(X_train), "n_test": len(X_test)}
        logger.info(f"  {target_col} forecast RMSE: {rmse:.4f}")

        # Save
        model_card = create_model_card(
            name=filename.replace(".pkl", ""),
            model_type="RandomForestRegressor",
            args=training_args,
            feature_list=forecast_features,
            metrics=metrics[target_col],
        )
        save_model_artifact(model, models_dir / filename, model_card)

    return metrics


def train_anomaly_detector(
    df: pd.DataFrame,
    models_dir: Path,
    seed: int,
) -> dict:
    """
    Train IsolationForest anomaly detector.

    Args:
        df: Featurized DataFrame.
        models_dir: Directory to save models.
        seed: Random seed.

    Returns:
        Dictionary with training info.
    """
    logger.info("Training IsolationForest anomaly detector...")
    metrics: dict = {}
    training_args = {"n_estimators": 200, "contamination": 0.01, "seed": seed}

    missing_features = [f for f in ANOM_FEATURES if f not in df.columns]
    if missing_features:
        logger.warning(
            f"Skipping anomaly training - missing features: {missing_features}")
        return metrics

    if len(df) < 100:
        logger.warning("Skipping anomaly training - too few rows (<100)")
        return metrics

    X = df[ANOM_FEATURES]
    model = IsolationForest(
        n_estimators=200, contamination=0.01, random_state=seed
    )
    model.fit(X)

    metrics["anomaly"] = {"n_samples": len(X), "contamination": 0.01}
    logger.info(f"  Trained on {len(X)} samples")

    model_card = create_model_card(
        name="anomaly_if",
        model_type="IsolationForest",
        args=training_args,
        feature_list=ANOM_FEATURES,
        metrics=metrics["anomaly"],
    )
    save_model_artifact(model, models_dir / "anomaly_if.pkl", model_card)

    return metrics


def train_classifier(
    df: pd.DataFrame,
    models_dir: Path,
    test_size: float,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """
    Train RandomForest classifier for water quality classification.

    Args:
        df: Featurized DataFrame.
        models_dir: Directory to save models.
        test_size: Fraction for test split.
        seed: Random seed.

    Returns:
        Tuple of (classification metrics dict, labeled DataFrame).
    """
    logger.info("Training RandomForest classifier...")
    training_args = {
        "n_estimators": 300,
        "max_depth": 14,
        "class_weight": {0: 1, 1: 2, 2: 10},
        "test_size": test_size,
        "seed": seed,
    }

    def label_row(r: pd.Series) -> int:
        """
        Conservative rule-based labels.

        Thresholds based on typical aquaculture standards.
        Adjust for species-specific requirements.
        """
        ph = r["ph"]
        tds = r["tds"]
        temp = r.get("temp", 25)  # Default if not present
        turbidity = r.get("turbidity", 0)  # Default if not present

        # Danger thresholds (critical values)
        if (
            ph < 6
            or ph > 9
            or temp < 15
            or temp > 34
            or turbidity > 150
            or tds > 1000
        ):
            return 2  # Danger

        # Stress thresholds (marginal values)
        if (
            (6 <= ph < 6.5)
            or (8.5 < ph <= 9)
            or tds > 500
            or turbidity > 50
        ):
            return 1  # Stress

        return 0  # Safe

    df = df.copy()
    df["label"] = df.apply(label_row, axis=1)

    # Log label distribution
    label_names = {0: "Safe", 1: "Stress", 2: "Danger"}
    for label_int, label_name in label_names.items():
        count = (df["label"] == label_int).sum()
        pct = 100 * count / len(df)
        logger.info(f"  Label {label_name}: {count} ({pct:.1f}%)")

    X = df[CLF_FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        class_weight={0: 1, 1: 2, 2: 10},
        random_state=seed,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    metrics = {
        "accuracy": report["accuracy"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_distribution": y.value_counts().to_dict(),
    }

    # Save model
    model_card = create_model_card(
        name="classifier_rf",
        model_type="RandomForestClassifier",
        args=training_args,
        feature_list=CLF_FEATURES,
        metrics=metrics,
    )
    save_model_artifact(model, models_dir / "classifier_rf.pkl", model_card)

    # Save feature list for backend
    feature_list_path = models_dir / "feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump(CLF_FEATURES, f)
    logger.info(f"✔ Saved feature list: {feature_list_path}")

    # Save labeled data for evaluation and reference
    # This creates features_labeled.csv with all features + label column
    return metrics, df  # Return df with labels for saving


def save_scaler_and_features(
    df: pd.DataFrame,
    models_dir: Path,
) -> None:
    """
    Save ammonia scaler for feature normalization.

    Args:
        df: Featurized DataFrame with ammonia_risk_raw.
        models_dir: Directory to save scaler.
    """
    if "ammonia_risk_raw" not in df.columns:
        logger.warning("Skipping scaler save - ammonia_risk_raw not found")
        return

    scaler = MinMaxScaler()
    df["ammonia_risk"] = scaler.fit_transform(df[["ammonia_risk_raw"]])
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")
    logger.info(f"✔ Saved ammonia scaler: {models_dir / 'feature_scaler.pkl'}")


@app.command()
def train(
    data_csv: Path = typer.Option(
        DEFAULT_DATA_DIR / "raw_combined.csv",
        "--data-csv",
        "-d",
        help="Path to the input CSV file.",
    ),
    models_dir: Path = typer.Option(
        DEFAULT_MODELS_DIR,
        "--models-dir",
        "-m",
        help="Directory to save trained models.",
    ),
    test_size: float = typer.Option(
        DEFAULT_TEST_SIZE,
        "--test-size",
        "-t",
        help="Fraction of data for testing.",
    ),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
    skip_virtual: bool = typer.Option(
        False,
        "--skip-virt",
        help="Skip training virtual sensor models.",
    ),
    skip_anomaly: bool = typer.Option(
        False,
        "--skip-anom",
        help="Skip training anomaly detector.",
    ),
    save_metrics: bool = typer.Option(
        True,
        "--save-metrics/--no-save-metrics",
        help="Save training metrics to JSON files.",
    ),
    min_rows: int = typer.Option(
        50,
        "--min-rows",
        help="Minimum rows required for training.",
    ),
) -> None:
    """
    Train all classical ML models for water quality prediction.

    This includes:
    - Virtual sensor regressors (temp, DO, turbidity)
    - RF forecasters (pH, TDS)
    - Anomaly detector (IsolationForest)
    - Multi-class classifier (Safe/Stress/Danger)
    """
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    set_seeds(seed)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate data
    try:
        df = load_data(data_csv)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    if len(df) < min_rows:
        logger.error(f"Insufficient data: {len(df)} rows < {min_rows} minimum")
        raise typer.Exit(code=1)

    # Clean and featurize
    df = clean_data(df)
    if len(df) < min_rows:
        logger.error(
            f"Insufficient data after cleaning: {len(df)} rows < {min_rows} minimum")
        raise typer.Exit(code=1)

    df = featurize(df)

    # Save scaler
    save_scaler_and_features(df, models_dir)

    # Save cleaned CSV
    cleaned_path = data_csv.parent / "cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    logger.info(f"✔ Saved cleaned data: {cleaned_path}")

    # Track all metrics
    all_metrics: dict = {
        "training_date": datetime.now().isoformat(),
        "data_csv": str(data_csv),
        "n_samples_raw": len(df),
        "seed": seed,
        "test_size": test_size,
        "models": {},
    }

    # Train models
    if not skip_virtual:
        virt_metrics = train_virtual_sensors(df, models_dir, test_size, seed)
        all_metrics["models"]["virtual_sensors"] = virt_metrics

    forecast_metrics = train_forecasters(df, models_dir, test_size, seed)
    all_metrics["models"]["forecasters"] = forecast_metrics

    if not skip_anomaly:
        anom_metrics = train_anomaly_detector(df, models_dir, seed)
        all_metrics["models"]["anomaly"] = anom_metrics

    clf_metrics, df_labeled = train_classifier(df, models_dir, test_size, seed)
    all_metrics["models"]["classifier"] = clf_metrics

    # Save labeled data (features + labels) for evaluation
    labeled_path = data_csv.parent / "features_labeled.csv"
    df_labeled.to_csv(labeled_path, index=False)
    logger.info(f"✔ Saved labeled data: {labeled_path}")

    # Save aggregated metrics
    if save_metrics:
        metrics_path = models_dir / "metrics_training.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info(f"✔ Saved training metrics: {metrics_path}")

    logger.info("=" * 60)
    logger.info("✔ All classical models trained and saved!")
    logger.info("=" * 60)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
