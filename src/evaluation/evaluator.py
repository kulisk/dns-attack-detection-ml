"""
Comprehensive evaluation module for DNS attack detection models.

Computes accuracy, precision, recall, F1, ROC-AUC, and produces
confusion-matrix and ROC-curve plots saved to disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.models.base_detector import BaseDetector
from src.utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="colorblind")


class Evaluator:
    """Evaluate a trained detector on labelled data.

    Args:
        class_names: Ordered list of class label strings.
        output_dir: Directory for saving metrics JSON and plot PNGs.
        average: Averaging strategy for multi-class metrics.
    """

    def __init__(
        self,
        class_names: list[str],
        output_dir: str = "reports",
        average: str = "weighted",
        dpi: int = 150,
    ) -> None:
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.average = average
        self.dpi = dpi

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        model: BaseDetector,
        X: np.ndarray,
        y_true: np.ndarray,
        split: str = "test",
    ) -> dict:
        """Run a full evaluation pass.

        Args:
            model: Fitted detector.
            X: Feature matrix.
            y_true: True integer labels.
            split: Tag for filenames (e.g. ``"test"``, ``"val"``).

        Returns:
            Metrics dictionary.
        """
        logger.info(f"Evaluating {model.name} on {split} set …")
        y_pred = model.predict(X)
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            y_proba = None

        metrics = self._compute_metrics(y_true, y_pred, y_proba)
        metrics["model"] = model.name
        metrics["split"] = split

        # Save metrics to JSON
        metrics_path = self.output_dir / f"{model.name}_{split}_metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(
                {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
                fh,
                indent=2,
            )

        # Plots
        self._plot_confusion_matrix(y_true, y_pred, model.name, split)
        if y_proba is not None:
            self._plot_roc_curves(y_true, y_proba, model.name, split)
        if hasattr(model, "feature_importances") and model.feature_names:
            self._plot_feature_importance(model, split)

        logger.info(
            f"Evaluation complete for {model.name}",
            extra={
                "accuracy": round(metrics["accuracy"], 4),
                "f1_weighted": round(metrics["f1_weighted"], 4),
                "roc_auc": round(metrics.get("roc_auc", 0.0), 4),
            },
        )
        return metrics

    def compare_models(
        self,
        results: list[dict],
        output_filename: str = "model_comparison.png",
    ) -> None:
        """Plot a side-by-side bar chart comparing multiple models.

        Args:
            results: List of metrics dicts from :meth:`evaluate`.
            output_filename: Output plot filename.
        """
        import pandas as pd
        metric_cols = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
        rows = []
        for r in results:
            rows.append({
                "model": r.get("model", "unknown"),
                **{m: r.get(m, 0.0) for m in metric_cols},
            })
        df = pd.DataFrame(rows).set_index("model")

        fig, ax = plt.subplots(figsize=(10, 5))
        df.plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Model Comparison", fontsize=14)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        path = self.output_dir / output_filename
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        logger.info("Model comparison plot saved", extra={"path": str(path)})

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
    ) -> dict:
        n_classes = len(self.class_names)
        roc_auc = 0.0
        if y_proba is not None and n_classes > 1:
            try:
                if n_classes == 2:
                    roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    roc_auc = float(
                        roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="weighted"
                        )
                    )
            except Exception as exc:
                logger.warning("ROC-AUC computation failed", extra={"error": str(exc)})

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred,
                target_names=self.class_names,
                zero_division=0,
                output_dict=True,
            ),
        }

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        split: str,
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)
        n = len(self.class_names)
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names,
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
        ax.set_title(f"Confusion Matrix – {model_name} ({split})", fontsize=13)
        plt.tight_layout()
        path = self.output_dir / f"{model_name}_{split}_confusion_matrix.png"
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        logger.info("Confusion matrix saved", extra={"path": str(path)})

    def _plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        split: str,
    ) -> None:
        n_classes = len(self.class_names)
        fig, ax = plt.subplots(figsize=(9, 7))

        if n_classes == 2:
            RocCurveDisplay.from_predictions(y_true, y_proba[:, 1], ax=ax, name=model_name)
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            for i, cls_name in enumerate(self.class_names):
                if y_bin[:, i].sum() == 0:
                    continue
                RocCurveDisplay.from_predictions(
                    y_bin[:, i],
                    y_proba[:, i],
                    ax=ax,
                    name=cls_name,
                    alpha=0.7,
                )

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_title(f"ROC Curves – {model_name} ({split})", fontsize=13)
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        path = self.output_dir / f"{model_name}_{split}_roc.png"
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        logger.info("ROC curves saved", extra={"path": str(path)})

    def _plot_feature_importance(
        self, model: BaseDetector, split: str, top_n: int = 25
    ) -> None:
        importances = model.feature_importances  # type: ignore[attr-defined]
        names = model.feature_names or [f"f{i}" for i in range(len(importances))]
        paired = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:top_n]
        names_top, imps_top = zip(*paired)

        fig, ax = plt.subplots(figsize=(9, max(5, top_n // 2)))
        ax.barh(names_top[::-1], imps_top[::-1], color="steelblue", edgecolor="black")
        ax.set_title(f"Feature Importances – {model.name} ({split})", fontsize=13)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        path = self.output_dir / f"{model.name}_{split}_feature_importance.png"
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        logger.info("Feature importance plot saved", extra={"path": str(path)})

    def _plot_anomaly_scores(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        model_name: str,
        threshold: float,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        benign_scores = scores[y_true == 0]
        attack_scores = scores[y_true == 1]
        ax.hist(benign_scores, bins=50, alpha=0.6, label="Benign", color="green")
        ax.hist(attack_scores, bins=50, alpha=0.6, label="Attack", color="red")
        ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.4f}")
        ax.set_title(f"Anomaly Score Distribution – {model_name}")
        ax.set_xlabel("Anomaly Score")
        ax.legend()
        plt.tight_layout()
        path = self.output_dir / f"{model_name}_anomaly_scores.png"
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        logger.info("Anomaly score plot saved", extra={"path": str(path)})
