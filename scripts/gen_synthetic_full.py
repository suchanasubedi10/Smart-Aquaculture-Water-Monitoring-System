#!/usr/bin/env python3
"""
Synthetic Water Quality Data Generator.

Generates realistic synthetic sensor data for water quality monitoring systems.
Supports configurable sample count, frequency, anomaly injection, and reproducible seeding.

Commit message: "scripts: make synthetic data generator CLI, add summary and tests"
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Generate synthetic water quality data for training and testing.")


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
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


def generate_synthetic_data(
    n_samples: int = 2000,
    freq: str = "5min",
    seed: int = 42,
    start_date: str = "2024-01-01 00:00",
) -> pd.DataFrame:
    """
    Generate synthetic water quality sensor data.

    Args:
        n_samples: Number of samples to generate.
        freq: Pandas frequency string for timestamp intervals (e.g., '5min', '1h').
        seed: Random seed for reproducibility.
        start_date: Start datetime string for the time series.

    Returns:
        DataFrame with columns: timestamp, ph, tds, temp, turbidity, do.
    """
    set_seeds(seed)

    # Generate datetime range
    rng = pd.date_range(start_date, periods=n_samples, freq=freq)
    n = len(rng)

    logger.info(
        f"Generating {n} samples with frequency '{freq}' starting from {start_date}")

    # Simulate realistic pH wave (circadian pattern + noise)
    ph_base = 7 + 0.7 * np.sin(np.arange(n) / 48) + \
        np.random.normal(0, 0.12, n)

    # TDS fluctuates widely
    tds_base = 300 + 60 * np.random.randn(n)

    # Temperature follows day-night cycle + correlation with TDS
    temp = (
        24
        + 3 * np.sin(np.arange(n) / 24)
        + 0.003 * tds_base
        + np.random.normal(0, 0.4, n)
    )

    # Turbidity depends on random spikes + TDS
    turbidity = np.clip(
        30 + 0.05 * tds_base + 20 * np.random.randn(n),
        0,
        400,
    )

    # DO decreases with temperature & turbidity
    do = np.clip(
        9 - 0.12 * temp - 0.01 * turbidity + 0.005 *
        (800 - tds_base) + np.random.normal(0, 0.3, n),
        0.2,
        12,
    )

    df = pd.DataFrame(
        {
            "timestamp": rng,
            "ph": np.clip(ph_base, 5.2, 9.6),
            "tds": np.clip(tds_base, 20, 2000),
            "temp": np.clip(temp, 10, 36),
            "turbidity": turbidity,
            "do": do,
        }
    )

    return df


def inject_anomalies(
    df: pd.DataFrame,
    n_anomalies: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Inject anomalies into the dataset.

    Args:
        df: Input DataFrame with water quality data.
        n_anomalies: Number of anomalies to inject.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with anomalies injected.
    """
    if n_anomalies <= 0:
        return df

    set_seeds(seed + 1000)  # Offset seed for anomaly injection

    df = df.copy()
    n_rows = len(df)

    if n_anomalies > n_rows:
        logger.warning(
            f"Requested {n_anomalies} anomalies but only {n_rows} rows. Capping.")
        n_anomalies = n_rows

    # Select random indices for anomalies
    anomaly_indices = np.random.choice(n_rows, size=n_anomalies, replace=False)

    logger.info(f"Injecting {n_anomalies} anomalies at random positions")

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(
            ["ph_spike", "tds_spike", "do_drop", "multi"])

        if anomaly_type == "ph_spike":
            # Extreme pH values
            df.loc[idx, "ph"] = np.random.choice([4.5, 10.5])
        elif anomaly_type == "tds_spike":
            # Very high TDS
            df.loc[idx, "tds"] = np.random.uniform(1500, 2500)
        elif anomaly_type == "do_drop":
            # Very low dissolved oxygen
            df.loc[idx, "do"] = np.random.uniform(0.1, 1.0)
        else:  # multi
            # Multiple anomalies at once
            df.loc[idx, "ph"] = np.random.uniform(5.0, 5.5)
            df.loc[idx, "tds"] = np.random.uniform(1200, 1800)
            df.loc[idx, "do"] = np.random.uniform(1.0, 2.0)

    return df


def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the generated data.

    Args:
        df: Input DataFrame with water quality data.

    Returns:
        Dictionary with mean, std, min, max for each numeric column.
    """
    numeric_cols = ["ph", "tds", "temp", "turbidity", "do"]
    summary: dict = {
        "generated_at": datetime.now().isoformat(),
        "n_samples": len(df),
        "columns": {},
    }

    for col in numeric_cols:
        if col in df.columns:
            summary["columns"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }

    return summary


@app.command()
def generate(
    out_dir: Path = typer.Option(
        Path("data"),
        "--out-dir",
        "-o",
        help="Output directory for generated files.",
    ),
    n_samples: int = typer.Option(
        2000,
        "--n-samples",
        "-n",
        help="Number of samples to generate.",
    ),
    freq: str = typer.Option(
        "5min",
        "--freq",
        "-f",
        help="Pandas frequency string for timestamp intervals.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
    add_anomalies: int = typer.Option(
        0,
        "--add-anomalies",
        "-a",
        help="Number of anomalies to inject into the data.",
    ),
    start_date: str = typer.Option(
        "2024-01-01 00:00",
        "--start-date",
        help="Start datetime for the time series.",
    ),
) -> None:
    """
    Generate synthetic water quality data and save to CSV.

    Outputs:
        - raw_combined.csv: The generated dataset
        - sample_summary.json: Summary statistics
    """
    logger.info(f"Starting synthetic data generation with seed={seed}")

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    df = generate_synthetic_data(
        n_samples=n_samples,
        freq=freq,
        seed=seed,
        start_date=start_date,
    )

    # Inject anomalies if requested
    if add_anomalies > 0:
        df = inject_anomalies(df, n_anomalies=add_anomalies, seed=seed)

    # Save CSV
    csv_path = out_dir / "raw_combined.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✔ Generated dataset → {csv_path}")
    logger.info(f"  Rows: {len(df)}")

    # Compute and save summary
    summary = compute_summary_statistics(df)
    summary["seed"] = seed
    summary["freq"] = freq
    summary["anomalies_injected"] = add_anomalies

    summary_path = out_dir / "sample_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✔ Saved summary → {summary_path}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
