#!/usr/bin/env python3
"""
PyTorch LSTM Training Script for Water Quality Forecasting.

Trains LSTM models for pH and TDS forecasting with checkpointing,
early stopping, and learning rate scheduling.

Commit message: "scripts: harden PyTorch LSTM trainer, add checkpoints & model card"
"""

from __future__ import annotations

import json
import logging
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import typer
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Train PyTorch LSTM models for water quality forecasting.")

# Default configuration from environment
DEFAULT_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DEFAULT_MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
DEFAULT_SEQ_LEN = int(os.getenv("LSTM_SEQ_LEN", "20"))
DEFAULT_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "64"))
DEFAULT_EPOCHS = int(os.getenv("LSTM_EPOCHS", "40"))
DEFAULT_LR = float(os.getenv("LSTM_LR", "0.001"))
DEFAULT_HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "64"))
DEFAULT_PATIENCE = int(os.getenv("LSTM_PATIENCE", "5"))
DEFAULT_SEED = int(os.getenv("RANDOM_SEED", "42"))


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug(f"Random seeds set to {seed}")


def get_device(device_str: str) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device_str: Device string ('auto', 'cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        torch.device object.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class WaterQualityDataset(Dataset):
    """
    PyTorch Dataset for water quality time series data.

    Provides sequences of (input_seq, target) pairs for LSTM training.
    """

    def __init__(self, data: np.ndarray, seq_len: int = 20) -> None:
        """
        Initialize the dataset.

        Args:
            data: Numpy array of shape (n_samples, n_features).
            seq_len: Number of time steps in each input sequence.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        if len(data) <= seq_len:
            raise ValueError(
                f"Data length ({len(data)}) must be > seq_len ({seq_len})")

        self.data = data.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        """Return the number of samples."""
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (input_sequence, target) tensors.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        x = self.data[idx: idx + self.seq_len]  # (seq_len, n_features)
        y = self.data[idx + self.seq_len]  # (n_features,)
        return torch.from_numpy(x), torch.from_numpy(y)


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.

    Architecture:
    - LSTM layers with configurable hidden size and depth
    - Fully connected layers for output projection
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units in LSTM.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between LSTM layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Output tensor of shape (batch,).
        """
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last = out[:, -1, :]  # Take last time step
        y = self.fc(last)  # (batch, 1)
        return y.squeeze(-1)  # (batch,)


class TrainingHistory:
    """Container for training history and metrics."""

    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.epochs: list[int] = []
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0

    def update(self, epoch: int, train_loss: float, val_loss: float) -> bool:
        """
        Update history and check if this is the best epoch.

        Returns:
            True if this is the best epoch so far.
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        is_best = val_loss < self.best_val_loss - 1e-6
        if is_best:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
        return is_best

    def to_dict(self) -> dict:
        """Convert history to dictionary."""
        return {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }


def train_one_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_idx: int,
    save_path: Path,
    device: torch.device,
    hidden_size: int = 64,
    num_layers: int = 1,
    epochs: int = 40,
    lr: float = 1e-3,
    patience: int = 5,
    grad_clip: Optional[float] = None,
    use_scheduler: bool = True,
) -> tuple[LSTMModel, TrainingHistory]:
    """
    Train a single LSTM model.

    Args:
        train_loader: Training data loader.
        val_loader: Validation data loader.
        target_idx: Index of target variable in output (0=pH, 1=TDS).
        save_path: Path to save the best model checkpoint.
        device: Torch device.
        hidden_size: LSTM hidden size.
        num_layers: Number of LSTM layers.
        epochs: Maximum training epochs.
        lr: Initial learning rate.
        patience: Early stopping patience.
        grad_clip: Gradient clipping max norm (None to disable).
        use_scheduler: Whether to use learning rate scheduler.

    Returns:
        Tuple of (trained model, training history).
    """
    model = LSTMModel(input_size=2, hidden_size=hidden_size,
                      num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6
        )

    history = TrainingHistory()
    wait = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_losses: list[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb[:, target_idx].to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses: list[float] = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb[:, target_idx].to(device)
                out = model(xb)
                val_losses.append(criterion(out, yb).item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[target={target_idx}] Epoch {epoch:02d}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  lr={current_lr:.2e}"
        )

        # Check for improvement
        is_best = history.update(epoch, train_loss, val_loss)

        if is_best:
            wait = 0
            torch.save(model.state_dict(), save_path)
            logger.debug(f"Checkpoint saved to {save_path}")
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    f"[target={target_idx}] Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history


def create_model_card(
    name: str,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    training_args: dict,
    history: TrainingHistory,
    scaler_path: str,
) -> dict:
    """
    Create a model card with LSTM-specific metadata.

    Args:
        name: Model name.
        seq_len: Sequence length used for training.
        hidden_size: LSTM hidden size.
        num_layers: Number of LSTM layers.
        training_args: Training arguments.
        history: Training history.
        scaler_path: Path to the scaler file.

    Returns:
        Model card dictionary.
    """
    return {
        "name": name,
        "version": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "model_type": "LSTMModel",
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "scaler_path": scaler_path,
        "training_args": training_args,
        "metrics_summary": {
            "best_val_loss": history.best_val_loss,
            "best_epoch": history.best_epoch,
            "total_epochs": len(history.epochs),
        },
    }


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
    seq_len: int = typer.Option(
        DEFAULT_SEQ_LEN,
        "--seq-len",
        help="Number of past time steps for input sequence.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        "-b",
        help="Training batch size.",
    ),
    epochs: int = typer.Option(
        DEFAULT_EPOCHS,
        "--epochs",
        "-e",
        help="Maximum training epochs.",
    ),
    lr: float = typer.Option(
        DEFAULT_LR,
        "--lr",
        help="Initial learning rate.",
    ),
    hidden_size: int = typer.Option(
        DEFAULT_HIDDEN_SIZE,
        "--hidden-size",
        help="LSTM hidden layer size.",
    ),
    num_layers: int = typer.Option(
        1,
        "--num-layers",
        help="Number of LSTM layers.",
    ),
    patience: int = typer.Option(
        DEFAULT_PATIENCE,
        "--patience",
        help="Early stopping patience.",
    ),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device to use (auto, cpu, cuda).",
    ),
    grad_clip: Optional[float] = typer.Option(
        None,
        "--grad-clip",
        help="Gradient clipping max norm (None to disable).",
    ),
    no_scheduler: bool = typer.Option(
        False,
        "--no-scheduler",
        help="Disable learning rate scheduler.",
    ),
) -> None:
    """
    Train PyTorch LSTM models for pH and TDS forecasting.

    Trains two separate LSTM models:
    - pH forecaster
    - TDS forecaster

    Saves model checkpoints, scalers, model cards, and training history.
    """
    logger.info("=" * 60)
    logger.info("Starting LSTM training")
    logger.info("=" * 60)

    # Setup
    set_seeds(seed)
    device_obj = get_device(device)
    logger.info(f"Using device: {device_obj}")

    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if not data_csv.exists():
        logger.error(f"Data file not found: {data_csv}")
        raise typer.Exit(code=1)

    df = pd.read_csv(data_csv, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required_cols = {"ph", "tds"}
    if not required_cols.issubset(df.columns):
        logger.error(f"CSV must contain columns: {required_cols}")
        raise typer.Exit(code=1)

    arr = df[["ph", "tds"]].values.astype(np.float32)
    logger.info(f"Loaded {len(arr)} samples")

    # Scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)

    scaler_path = models_dir / "lstm_scaler_pytorch.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"✔ Saved scaler: {scaler_path}")

    # Create dataset
    try:
        dataset = WaterQualityDataset(scaled, seq_len=seq_len)
    except ValueError as e:
        logger.error(f"Dataset creation failed: {e}")
        raise typer.Exit(code=1)

    if len(dataset) < 10:
        logger.error(f"Not enough data to train. Need > {seq_len} samples.")
        raise typer.Exit(code=1)

    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, drop_last=False)

    logger.info(
        f"Dataset: {len(dataset)} samples, train: {train_size}, val: {val_size}")

    training_args = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "patience": patience,
        "seed": seed,
        "grad_clip": grad_clip,
        "use_scheduler": not no_scheduler,
    }

    all_histories: dict = {
        "training_date": datetime.now().isoformat(),
        "training_args": training_args,
        "models": {},
    }

    # Train pH model
    logger.info("Training pH LSTM model (target index 0)...")
    ph_model_path = models_dir / "forecast_ph_lstm_pt.pth"
    _, ph_history = train_one_model(
        train_loader=train_loader,
        val_loader=val_loader,
        target_idx=0,
        save_path=ph_model_path,
        device=device_obj,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        patience=patience,
        grad_clip=grad_clip,
        use_scheduler=not no_scheduler,
    )
    logger.info(f"✔ Saved pH model: {ph_model_path}")

    # Save pH model card
    ph_card = create_model_card(
        name="forecast_ph_lstm_pt",
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        training_args=training_args,
        history=ph_history,
        scaler_path=str(scaler_path.name),
    )
    with open(ph_model_path.with_suffix(".model_card.json"), "w") as f:
        json.dump(ph_card, f, indent=2)

    all_histories["models"]["ph"] = ph_history.to_dict()

    # Train TDS model
    logger.info("Training TDS LSTM model (target index 1)...")
    tds_model_path = models_dir / "forecast_tds_lstm_pt.pth"
    _, tds_history = train_one_model(
        train_loader=train_loader,
        val_loader=val_loader,
        target_idx=1,
        save_path=tds_model_path,
        device=device_obj,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        patience=patience,
        grad_clip=grad_clip,
        use_scheduler=not no_scheduler,
    )
    logger.info(f"✔ Saved TDS model: {tds_model_path}")

    # Save TDS model card
    tds_card = create_model_card(
        name="forecast_tds_lstm_pt",
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        training_args=training_args,
        history=tds_history,
        scaler_path=str(scaler_path.name),
    )
    with open(tds_model_path.with_suffix(".model_card.json"), "w") as f:
        json.dump(tds_card, f, indent=2)

    all_histories["models"]["tds"] = tds_history.to_dict()

    # Save combined training history
    history_path = models_dir / "metrics_lstm_training.json"
    with open(history_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    logger.info(f"✔ Saved training history: {history_path}")

    logger.info("=" * 60)
    logger.info(f"✔ LSTM training complete. Models saved in {models_dir}")
    logger.info("=" * 60)


def inference_from_array(
    model_path: Path,
    scaler_path: Path,
    seq_array: np.ndarray,
    steps: int = 5,
    target_idx: int = 0,
    hidden_size: int = 64,
    num_layers: int = 1,
    device: str = "cpu",
) -> list[float]:
    """
    Run inference from a numpy array of recent readings.

    Args:
        model_path: Path to the saved model weights.
        scaler_path: Path to the fitted scaler.
        seq_array: Array of shape (seq_len, 2) with [ph, tds] values.
        steps: Number of future steps to predict.
        target_idx: Which feature to predict (0=pH, 1=TDS).
        hidden_size: Model hidden size (must match training).
        num_layers: Number of LSTM layers (must match training).
        device: Device string.

    Returns:
        List of predicted values in original scale.
    """
    device_obj = torch.device(device)

    # Load model
    model = LSTMModel(input_size=2, hidden_size=hidden_size,
                      num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device_obj))
    model.to(device_obj)
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Prepare input
    seq_len = len(seq_array)
    scaled = scaler.transform(seq_array.astype(np.float32))
    seq_scaled = scaled.copy()

    predictions: list[float] = []

    for _ in range(steps):
        x = torch.from_numpy(seq_scaled.reshape(
            1, seq_len, 2)).float().to(device_obj)

        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy().item()

        # Inverse transform
        if target_idx == 0:
            inv = scaler.inverse_transform([[pred_scaled, 0.0]])[0][0]
        else:
            inv = scaler.inverse_transform([[0.0, pred_scaled]])[0][1]

        predictions.append(float(inv))

        # Update sequence for next prediction
        new_row = np.array(
            [pred_scaled if i == target_idx else seq_scaled[-1, i] for i in range(2)])
        seq_scaled = np.vstack([seq_scaled[1:], new_row.reshape(1, 2)])

    return predictions


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
