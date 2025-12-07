#!/usr/bin/env python3
"""
Evaluate the trained water quality classifier.

Generates classification report, confusion matrix, and per-class metrics.

Commit message: "scripts: refactor evaluate_classifier with CLI and metrics export"
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Evaluate the water quality classifier.")

# Default paths from environment
DEFAULT_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DEFAULT_MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

# Label names for pretty printing
LABEL_NAMES = {0: "Safe", 1: "Stress", 2: "Danger"}

# Classifier features (must match training)
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


def label_row(row: pd.Series) -> int:
    """
    Apply rule-based labeling to a data row.

    This should match the labeling logic used during training.
    Adjust thresholds based on species-specific requirements.

    Args:
        row: A row from the DataFrame with 'ph', 'tds', and optionally
             'temp' and 'turbidity' columns.

    Returns:
        Label integer: 0=Safe, 1=Stress, 2=Danger.
    """
    ph = row["ph"]
    tds = row["tds"]
    temp = row.get("temp", 25)  # Default if not present
    turbidity = row.get("turbidity", 0)  # Default if not present

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


def load_classifier(models_dir: Path) -> object:
    """
    Load the trained classifier model.

    Args:
        models_dir: Directory containing model files.

    Returns:
        Loaded classifier model.

    Raises:
        FileNotFoundError: If classifier file not found.
    """
    model_path = models_dir / "classifier_rf.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier not found: {model_path}")

    logger.info(f"Loading classifier from {model_path}")
    return joblib.load(model_path)


def load_and_validate_data(
    data_path: Path,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load data and validate required features exist.

    Args:
        data_path: Path to the CSV file.
        features: List of required feature names.

    Returns:
        Tuple of (DataFrame, list of missing features).

    Raises:
        FileNotFoundError: If data file not found.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing = [f for f in features if f not in df.columns]
    return df, missing


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with all metrics.
    """
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )

    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_w),
        "weighted_recall": float(recall_w),
        "weighted_f1": float(f1_w),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
        "class_distribution": {
            LABEL_NAMES[i]: int((y_true == i).sum())
            for i in range(3)
        },
    }


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
) -> None:
    """
    Print a formatted evaluation report to console.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        metrics: Pre-computed metrics dictionary.
    """
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)

    # Use sklearn's report for nice formatting
    report = classification_report(
        y_true,
        y_pred,
        target_names=[LABEL_NAMES[i] for i in range(3)],
        digits=3,
        zero_division=0,
    )
    print(report)

    logger.info("\nCONFUSION MATRIX")
    logger.info("-" * 40)
    cm = np.array(metrics["confusion_matrix"])
    print(f"{'':>10} {'Safe':>8} {'Stress':>8} {'Danger':>8}")
    for i, label in enumerate(["Safe", "Stress", "Danger"]):
        print(f"{label:>10} {cm[i, 0]:>8} {cm[i, 1]:>8} {cm[i, 2]:>8}")

    logger.info("\nCLASS DISTRIBUTION")
    logger.info("-" * 40)
    for label, count in metrics["class_distribution"].items():
        pct = 100 * count / sum(metrics["class_distribution"].values())
        print(f"  {label}: {count} ({pct:.1f}%)")


@app.command()
def evaluate(
    data_csv: Path = typer.Option(
        DEFAULT_DATA_DIR / "cleaned.csv",
        "--data-csv",
        "-d",
        help="Path to the input CSV file (cleaned or features_labeled).",
    ),
    models_dir: Path = typer.Option(
        DEFAULT_MODELS_DIR,
        "--models-dir",
        "-m",
        help="Directory containing trained models.",
    ),
    output_metrics: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save metrics JSON (optional).",
    ),
    use_existing_labels: bool = typer.Option(
        False,
        "--use-labels",
        "-l",
        help="Use existing 'label' column instead of rule-based labeling.",
    ),
    save_labeled: bool = typer.Option(
        False,
        "--save-labeled",
        help="Save labeled data to features_labeled.csv.",
    ),
) -> None:
    """
    Evaluate the trained water quality classifier.

    By default, applies rule-based labeling to create ground truth.
    Use --use-labels to evaluate against existing labels in the data.
    """
    logger.info("=" * 60)
    logger.info("Water Quality Classifier Evaluation")
    logger.info("=" * 60)

    # Load classifier
    try:
        clf = load_classifier(models_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    # Load data
    try:
        df, missing = load_and_validate_data(data_csv, CLF_FEATURES)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    if missing:
        logger.error(f"Missing required features: {missing}")
        logger.error(
            "Ensure the data has been featurized (run train_all_models.py first)")
        raise typer.Exit(code=1)

    logger.info(f"Loaded {len(df)} samples")

    # Create or use labels
    if use_existing_labels:
        if "label" not in df.columns:
            logger.error("--use-labels specified but 'label' column not found")
            raise typer.Exit(code=1)
        logger.info("Using existing labels from data")
    else:
        logger.info("Applying rule-based labeling...")
        df["label"] = df.apply(label_row, axis=1)

    # Save labeled data if requested
    if save_labeled:
        labeled_path = data_csv.parent / "features_labeled.csv"
        df.to_csv(labeled_path, index=False)
        logger.info(f"✔ Saved labeled data: {labeled_path}")

    # Extract features and labels
    X = df[CLF_FEATURES]
    y_true = df["label"].values

    # Make predictions
    logger.info("Running classifier predictions...")
    y_pred = clf.predict(X)

    # Compute metrics
    metrics = compute_evaluation_metrics(y_true, y_pred)
    metrics["evaluation_date"] = datetime.now().isoformat()
    metrics["data_file"] = str(data_csv)
    metrics["n_samples"] = len(df)
    metrics["used_existing_labels"] = use_existing_labels

    # Print report
    print_evaluation_report(y_true, y_pred, metrics)

    # Save metrics if requested
    if output_metrics:
        output_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✔ Saved metrics: {output_metrics}")
    else:
        # Default save location
        default_output = models_dir / "metrics_classifier_eval.json"
        with open(default_output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✔ Saved metrics: {default_output}")

    logger.info("=" * 60)
    logger.info(f"✔ Evaluation complete! Accuracy: {metrics['accuracy']:.3f}")
    logger.info("=" * 60)


@app.command()
def create_labels(
    data_csv: Path = typer.Option(
        DEFAULT_DATA_DIR / "cleaned.csv",
        "--data-csv",
        "-d",
        help="Path to the input CSV file.",
    ),
    output_csv: Path = typer.Option(
        DEFAULT_DATA_DIR / "features_labeled.csv",
        "--output",
        "-o",
        help="Path to save labeled CSV.",
    ),
) -> None:
    """
    Create rule-based labels for the dataset.

    Useful for bootstrapping labels when ground-truth labels are unavailable.
    Labels: 0=Safe, 1=Stress, 2=Danger based on pH, TDS, temp, turbidity thresholds.
    """
    logger.info("Creating rule-based labels...")

    if not data_csv.exists():
        logger.error(f"Data file not found: {data_csv}")
        raise typer.Exit(code=1)

    df = pd.read_csv(data_csv, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df)} samples from {data_csv}")

    # Apply labeling
    df["label"] = df.apply(label_row, axis=1)

    # Log distribution
    for label_int, label_name in LABEL_NAMES.items():
        count = (df["label"] == label_int).sum()
        pct = 100 * count / len(df)
        logger.info(f"  {label_name}: {count} ({pct:.1f}%)")

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"✔ Saved labeled data: {output_csv}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
