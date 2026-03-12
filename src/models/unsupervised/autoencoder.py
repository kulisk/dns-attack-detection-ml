"""
Autoencoder anomaly detector using PyTorch.

Trained solely on *benign* DNS traffic.  An unusually high
reconstruction error at inference time flags a potential attack.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_detector import BaseDetector
from src.utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────── Architecture ────────────────────────────

class _Autoencoder(nn.Module):
    """Symmetric encoder-decoder with bottleneck."""

    def __init__(self, input_dim: int, encoding_dims: list[int]) -> None:
        super().__init__()
        # Encoder
        enc_layers: list[nn.Module] = []
        in_d = input_dim
        for d in encoding_dims:
            enc_layers += [nn.Linear(in_d, d), nn.BatchNorm1d(d), nn.ReLU()]
            in_d = d
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers: list[nn.Module] = []
        decode_dims = list(reversed(encoding_dims[:-1])) + [input_dim]
        for d in decode_dims:
            dec_layers += [nn.Linear(in_d, d), nn.ReLU()]
            in_d = d
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decoder(self.encoder(x))


# ──────────────────────────── Detector ────────────────────────────────

class AutoencoderDetector(BaseDetector):
    """Autoencoder-based unsupervised anomaly detector.

    Args:
        input_dim: Feature dimension.
        encoding_dims: Hidden layer sizes (bottleneck = last element).
        learning_rate: Adam LR.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        anomaly_threshold_percentile: Percentile of training reconstruction
            errors used as the detection threshold.
        device: Compute device.
        model_dir: Persistence directory.
    """

    def __init__(
        self,
        input_dim: int = 50,
        encoding_dims: list[int] | None = None,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        patience: int = 10,
        anomaly_threshold_percentile: float = 95.0,
        device: Optional[str] = None,
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="autoencoder", model_dir=model_dir)
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims or [64, 32, 16]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._threshold: float = 0.0
        self._build_network()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "AutoencoderDetector":
        if X_train.shape[1] != self.input_dim:
            self.input_dim = X_train.shape[1]
            self._build_network()

        logger.info(
            "Training Autoencoder",
            extra={"n_samples": len(X_train), "input_dim": self.input_dim},
        )

        train_loader = self._make_loader(X_train, shuffle=True)
        val_loader = self._make_loader(X_val) if X_val is not None else None

        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            if val_loader:
                val_loss = self._eval_epoch(val_loader, criterion)
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve = 0
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    logger.info("Early stopping", extra={"epoch": epoch})
                    break

        if best_state:
            self._model.load_state_dict(best_state)

        # Compute threshold from training reconstruction errors
        train_errors = self._reconstruction_errors(X_train)
        self._threshold = float(np.percentile(train_errors, self.anomaly_threshold_percentile))
        logger.info(
            "Autoencoder training complete",
            extra={"threshold": round(self._threshold, 6)},
        )
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 0 (benign) or 1 (attack) based on reconstruction error."""
        errors = self._reconstruction_errors(X)
        return (errors > self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return [prob_benign, prob_attack] based on normalised reconstruction error."""
        errors = self._reconstruction_errors(X)
        prob_attack = np.clip(errors / (self._threshold * 2 + 1e-9), 0.0, 1.0)
        return np.column_stack([1.0 - prob_attack, prob_attack])

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw per-sample reconstruction MSE."""
        return self._reconstruction_errors(X)

    def get_params(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "encoding_dims": self.encoding_dims,
            "learning_rate": self.learning_rate,
            "anomaly_threshold_percentile": self.anomaly_threshold_percentile,
        }

    def save(self, filename: Optional[str] = None) -> str:
        path = self.model_dir / (filename or "autoencoder.pt")
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "params": self.get_params(),
                "threshold": self._threshold,
                "feature_names": self._feature_names,
            },
            path,
        )
        logger.info("Autoencoder saved", extra={"path": str(path)})
        return str(path)

    def load(self, filename: Optional[str] = None) -> "AutoencoderDetector":
        path = self.model_dir / (filename or "autoencoder.pt")
        payload = torch.load(path, map_location=self.device)
        params = payload["params"]
        self.input_dim = params["input_dim"]
        self.encoding_dims = params["encoding_dims"]
        self._build_network()
        self._model.load_state_dict(payload["model_state"])
        self._threshold = payload["threshold"]
        self._feature_names = payload.get("feature_names", [])
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_network(self) -> None:
        self._model = _Autoencoder(self.input_dim, self.encoding_dims).to(self.device)

    def _make_loader(self, X: np.ndarray, shuffle: bool = False) -> DataLoader:
        ds = TensorDataset(torch.FloatTensor(X))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(self, loader, optimizer, criterion) -> float:
        self._model.train()
        total = 0.0
        for (X_b,) in loader:
            X_b = X_b.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self._model(X_b), X_b)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(X_b)
        return total / len(loader.dataset)

    def _eval_epoch(self, loader, criterion) -> float:
        self._model.eval()
        total = 0.0
        with torch.no_grad():
            for (X_b,) in loader:
                X_b = X_b.to(self.device)
                total += criterion(self._model(X_b), X_b).item() * len(X_b)
        return total / len(loader.dataset)

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            reconstructed = self._model(tensor).cpu().numpy()
        return np.mean((X - reconstructed) ** 2, axis=1)
