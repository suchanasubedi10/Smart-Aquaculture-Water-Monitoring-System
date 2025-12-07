"""
Tests for LSTM Dataset class (train_lstm_pytorch.py).

Commit message: "scripts: harden PyTorch LSTM trainer, add checkpoints & model card"
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestWaterQualityDataset:
    """Tests for the WaterQualityDataset class."""

    def test_dataset_length(self) -> None:
        """Test that dataset length is correct."""
        from train_lstm_pytorch import WaterQualityDataset

        data = np.random.randn(100, 2).astype(np.float32)
        dataset = WaterQualityDataset(data, seq_len=20)

        assert len(dataset) == 80  # 100 - 20

    def test_dataset_returns_correct_shapes(self) -> None:
        """Test that dataset returns tensors with correct shapes."""
        from train_lstm_pytorch import WaterQualityDataset

        data = np.random.randn(50, 2).astype(np.float32)
        seq_len = 10
        dataset = WaterQualityDataset(data, seq_len=seq_len)

        x, y = dataset[0]

        assert x.shape == (seq_len, 2)
        assert y.shape == (2,)

    def test_dataset_returns_tensors(self) -> None:
        """Test that dataset returns torch tensors."""
        from train_lstm_pytorch import WaterQualityDataset

        data = np.random.randn(30, 2).astype(np.float32)
        dataset = WaterQualityDataset(data, seq_len=10)

        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_dataset_values_match_input(self) -> None:
        """Test that returned values match input data."""
        from train_lstm_pytorch import WaterQualityDataset

        data = np.arange(40).reshape(20, 2).astype(np.float32)
        dataset = WaterQualityDataset(data, seq_len=5)

        x, y = dataset[0]

        # x should be rows 0-4, y should be row 5
        np.testing.assert_array_equal(x.numpy(), data[0:5])
        np.testing.assert_array_equal(y.numpy(), data[5])

    def test_dataset_second_item(self) -> None:
        """Test that second item is offset correctly."""
        from train_lstm_pytorch import WaterQualityDataset

        data = np.arange(40).reshape(20, 2).astype(np.float32)
        dataset = WaterQualityDataset(data, seq_len=5)

        x, y = dataset[1]

        # x should be rows 1-5, y should be row 6
        np.testing.assert_array_equal(x.numpy(), data[1:6])
        np.testing.assert_array_equal(y.numpy(), data[6])

    def test_dataset_rejects_1d_array(self) -> None:
        """Test that 1D array is rejected."""
        from train_lstm_pytorch import WaterQualityDataset

        data = np.random.randn(50).astype(np.float32)

        with pytest.raises(ValueError, match="2D"):
            WaterQualityDataset(data, seq_len=10)

    def test_dataset_rejects_short_data(self) -> None:
        """Test that data shorter than seq_len is rejected."""
        from train_lstm_pytorch import WaterQualityDataset

        data = np.random.randn(10, 2).astype(np.float32)

        with pytest.raises(ValueError, match="must be >"):
            WaterQualityDataset(data, seq_len=15)

    def test_dataset_rejects_non_numpy(self) -> None:
        """Test that non-numpy input is rejected."""
        from train_lstm_pytorch import WaterQualityDataset

        data = [[1, 2], [3, 4], [5, 6]]

        with pytest.raises(TypeError, match="numpy array"):
            WaterQualityDataset(data, seq_len=2)  # type: ignore

    def test_dataset_index_out_of_range(self) -> None:
        """Test that out-of-range index raises error."""
        from train_lstm_pytorch import WaterQualityDataset

        data = np.random.randn(30, 2).astype(np.float32)
        dataset = WaterQualityDataset(data, seq_len=10)

        with pytest.raises(IndexError):
            _ = dataset[100]


class TestLSTMModel:
    """Tests for the LSTM model architecture."""

    def test_model_forward_shape(self) -> None:
        """Test that model forward pass produces correct shape."""
        from train_lstm_pytorch import LSTMModel

        model = LSTMModel(input_size=2, hidden_size=32, num_layers=1)
        x = torch.randn(16, 20, 2)  # batch=16, seq_len=20, features=2

        output = model(x)

        assert output.shape == (16,)

    def test_model_with_different_hidden_sizes(self) -> None:
        """Test model works with different hidden sizes."""
        from train_lstm_pytorch import LSTMModel

        for hidden_size in [16, 32, 64, 128]:
            model = LSTMModel(input_size=2, hidden_size=hidden_size)
            x = torch.randn(8, 10, 2)
            output = model(x)
            assert output.shape == (8,)

    def test_model_with_multiple_layers(self) -> None:
        """Test model works with multiple LSTM layers."""
        from train_lstm_pytorch import LSTMModel

        model = LSTMModel(input_size=2, hidden_size=32, num_layers=3)
        x = torch.randn(8, 10, 2)
        output = model(x)
        assert output.shape == (8,)


class TestTrainingHistory:
    """Tests for TrainingHistory class."""

    def test_history_tracks_losses(self) -> None:
        """Test that history tracks losses correctly."""
        from train_lstm_pytorch import TrainingHistory

        history = TrainingHistory()
        history.update(1, 0.5, 0.6)
        history.update(2, 0.4, 0.5)

        assert history.train_losses == [0.5, 0.4]
        assert history.val_losses == [0.6, 0.5]
        assert history.epochs == [1, 2]

    def test_history_tracks_best(self) -> None:
        """Test that history tracks best epoch."""
        from train_lstm_pytorch import TrainingHistory

        history = TrainingHistory()
        is_best1 = history.update(1, 0.5, 0.6)
        is_best2 = history.update(2, 0.4, 0.4)
        is_best3 = history.update(3, 0.3, 0.5)

        assert is_best1  # First is always best
        assert is_best2  # 0.4 < 0.6
        assert not is_best3  # 0.5 > 0.4

        assert history.best_epoch == 2
        assert history.best_val_loss == 0.4

    def test_history_to_dict(self) -> None:
        """Test that history converts to dict correctly."""
        from train_lstm_pytorch import TrainingHistory

        history = TrainingHistory()
        history.update(1, 0.5, 0.6)

        d = history.to_dict()

        assert "epochs" in d
        assert "train_losses" in d
        assert "val_losses" in d
        assert "best_epoch" in d
        assert "best_val_loss" in d
