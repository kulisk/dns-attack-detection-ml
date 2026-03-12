"""
Multi-Layer Perceptron (MLP) detector using PyTorch.

Implements early stopping, dropout regularisation, and supports both
binary and multi-class classification for DNS attack detection.
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

class _MLPNetwork(nn.Module):
    """Fully-connected MLP with batch normalisation and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        n_classes: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# ──────────────────────────── Detector ────────────────────────────────

class MLPDetector(BaseDetector):
    """PyTorch MLP multi-class DNS attack detector.

    Args:
        input_dim: Number of input features.
        n_classes: Number of target classes.
        hidden_layers: Neuron counts per hidden layer.
        dropout_rate: Dropout probability.
        learning_rate: Adam optimiser LR.
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        patience: Early-stopping patience (epochs without val improvement).
        device: ``"cuda"`` or ``"cpu"`` (auto-detects by default).
        model_dir: Persistence directory.
    """

    def __init__(
        self,
        input_dim: int = 50,
        n_classes: int = 8,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        patience: int = 10,
        device: Optional[str] = None,
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="mlp", model_dir=model_dir)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._classes: list[int] = list(range(n_classes))
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
    ) -> "MLPDetector":
        """Train the MLP with Adam + cross-entropy + early stopping."""
        # Rebuild if dimensions changed
        if X_train.shape[1] != self.input_dim:
            self.input_dim = X_train.shape[1]
            self._build_network()

        n_classes = len(np.unique(y_train))
        if n_classes != self.n_classes:
            self.n_classes = n_classes
            self._build_network()

        logger.info(
            "Training MLP",
            extra={
                "n_samples": len(X_train),
                "input_dim": self.input_dim,
                "n_classes": self.n_classes,
                "device": str(self.device),
            },
        )

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val) if X_val is not None else None

        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_val_loss = float("inf")
        no_improve = 0
        best_state = None

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)

            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader, criterion)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    best_state = {
                        k: v.clone() for k, v in self._model.state_dict().items()
                    }
                else:
                    no_improve += 1

                if epoch % 10 == 0:
                    logger.debug(
                        f"Epoch {epoch}/{self.epochs}",
                        extra={"train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)},
                    )

                if no_improve >= self.patience:
                    logger.info(
                        "Early stopping triggered",
                        extra={"epoch": epoch, "best_val_loss": round(best_val_loss, 4)},
                    )
                    break
            else:
                if epoch % 10 == 0:
                    logger.debug(f"Epoch {epoch}/{self.epochs}", extra={"train_loss": round(train_loss, 4)})

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._is_fitted = True
        logger.info("MLP training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def get_params(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
        }

    def save(self, filename: Optional[str] = None) -> str:
        import joblib
        path = self.model_dir / (filename or "mlp.pt")
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "params": self.get_params(),
                "feature_names": self._feature_names,
            },
            path,
        )
        logger.info("MLP saved", extra={"path": str(path)})
        return str(path)

    def load(self, filename: Optional[str] = None) -> "MLPDetector":
        path = self.model_dir / (filename or "mlp.pt")
        payload = torch.load(path, map_location=self.device)
        params = payload["params"]
        self.input_dim = params["input_dim"]
        self.n_classes = params["n_classes"]
        self._build_network()
        self._model.load_state_dict(payload["model_state"])
        self._feature_names = payload.get("feature_names", [])
        self._is_fitted = True
        logger.info("MLP loaded", extra={"path": str(path)})
        return self

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_network(self) -> None:
        self._model = _MLPNetwork(
            self.input_dim, self.hidden_layers, self.n_classes, self.dropout_rate
        ).to(self.device)

    def _make_loader(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        shuffle: bool = False,
    ) -> DataLoader:
        X_t = torch.FloatTensor(X)
        if y is not None:
            y_t = torch.LongTensor(y.astype(int))
            ds = TensorDataset(X_t, y_t)
        else:
            ds = TensorDataset(X_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(self, loader: DataLoader, optimizer, criterion) -> float:
        self._model.train()
        total_loss = 0.0
        for batch in loader:
            X_b, y_b = batch
            X_b, y_b = X_b.to(self.device), y_b.to(self.device)
            optimizer.zero_grad()
            logits = self._model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_b)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader, criterion) -> float:
        self._model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                X_b, y_b = batch
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                logits = self._model(X_b)
                loss = criterion(logits, y_b)
                total_loss += loss.item() * len(X_b)
        return total_loss / len(loader.dataset)
