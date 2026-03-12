"""
LSTM detector for sequential DNS traffic classification.

Models DNS queries as time-ordered sequences to capture temporal
patterns characteristic of tunneling, DGA, and botnet traffic.
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

class _LSTMNetwork(nn.Module):
    """Stacked LSTM with a classification head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        n_classes: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Use last time-step's hidden state
        last = out[:, -1, :]
        return self.classifier(last)


# ──────────────────────────── Detector ────────────────────────────────

class LSTMDetector(BaseDetector):
    """LSTM-based sequential DNS attack detector.

    Args:
        input_size: Feature dimension per time step.
        n_classes: Number of target classes.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        sequence_length: Number of DNS queries per sequence sample.
        dropout_rate: Dropout probability.
        learning_rate: Adam optimiser LR.
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        device: Compute device.
        model_dir: Persistence directory.
    """

    def __init__(
        self,
        input_size: int = 50,
        n_classes: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        sequence_length: int = 20,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 30,
        patience: int = 7,
        device: Optional[str] = None,
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="lstm", model_dir=model_dir)
        self.input_size = input_size
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._build_network()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMDetector":
        """Train using 3-D sequence arrays.

        Args:
            X_train: Shape ``(n_samples, seq_len, n_features)`` or
                ``(n_samples, n_features)`` (auto-reshaped).
            y_train: Integer class labels, shape ``(n_samples,)``.
        """
        X_train = self._ensure_3d(X_train)
        self.input_size = X_train.shape[2]
        self.sequence_length = X_train.shape[1]
        self.n_classes = len(np.unique(y_train))
        self._build_network()

        logger.info(
            "Training LSTM",
            extra={
                "n_samples": len(X_train),
                "seq_len": self.sequence_length,
                "input_size": self.input_size,
                "n_classes": self.n_classes,
            },
        )

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = (
            self._make_loader(self._ensure_3d(X_val), y_val)
            if X_val is not None else None
        )

        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

        best_val_loss = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader, criterion)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    logger.info("Early stopping", extra={"epoch": epoch})
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._is_fitted = True
        logger.info("LSTM training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._ensure_3d(X)
        self._model.eval()
        tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def get_params(self) -> dict:
        return {
            "input_size": self.input_size,
            "n_classes": self.n_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
        }

    def save(self, filename: Optional[str] = None) -> str:
        path = self.model_dir / (filename or "lstm.pt")
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "params": self.get_params(),
                "feature_names": self._feature_names,
            },
            path,
        )
        logger.info("LSTM saved", extra={"path": str(path)})
        return str(path)

    def load(self, filename: Optional[str] = None) -> "LSTMDetector":
        path = self.model_dir / (filename or "lstm.pt")
        payload = torch.load(path, map_location=self.device)
        params = payload["params"]
        for k, v in params.items():
            setattr(self, k, v)
        self._build_network()
        self._model.load_state_dict(payload["model_state"])
        self._feature_names = payload.get("feature_names", [])
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_network(self) -> None:
        self._model = _LSTMNetwork(
            self.input_size, self.hidden_size, self.num_layers,
            self.n_classes, self.dropout_rate
        ).to(self.device)

    def _ensure_3d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            # Treat each row as a sequence of length 1 with input_size features
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return X

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> DataLoader:
        ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y.astype(int)))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(self, loader, optimizer, criterion) -> float:
        self._model.train()
        total = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(self.device), y_b.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self._model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
            optimizer.step()
            total += loss.item() * len(X_b)
        return total / len(loader.dataset)

    def _eval_epoch(self, loader, criterion) -> float:
        self._model.eval()
        total = 0.0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                total += criterion(self._model(X_b), y_b).item() * len(X_b)
        return total / len(loader.dataset)
