"""
Full training run: supervised + unsupervised models, GPU-accelerated where possible.

Usage:
    python run_training.py                  # all models
    python run_training.py --supervised     # supervised only
    python run_training.py --unsupervised   # unsupervised only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import ModelTrainer
from src.evaluation import Evaluator
from src.utils import get_logger

logger = get_logger("run_training", log_file="logs/run_training.log")

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


def _device_info() -> str:
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU only"


def train_supervised(trainer: ModelTrainer) -> list[dict]:
    from src.models.supervised import (
        RandomForestDetector,
        XGBoostDetector,
        SVMDetector,
        MLPDetector,
        LSTMDetector,
        SemiSupervisedEnsembleDetector,
    )

    models = [
        ("random_forest", lambda: RandomForestDetector(model_dir=str(MODELS_DIR))),
        ("xgboost",       lambda: XGBoostDetector(model_dir=str(MODELS_DIR))),
        ("svm",           lambda: SVMDetector(model_dir=str(MODELS_DIR), max_train_samples=15000)),
        ("mlp",           lambda: MLPDetector(model_dir=str(MODELS_DIR))),
        ("lstm",          lambda: LSTMDetector(model_dir=str(MODELS_DIR))),
        ("ensemble_neural", lambda: SemiSupervisedEnsembleDetector(model_dir=str(MODELS_DIR))),
    ]

    results = []
    for name, factory in models:
        logger.info(f"{'='*55}")
        logger.info(f"  Training supervised: {name}")
        logger.info(f"{'='*55}")
        t0 = time.perf_counter()
        try:
            model = factory()
            result = trainer.run(model, use_smote=False)
            elapsed = time.perf_counter() - t0
            result["metrics"]["training_time_s"] = round(elapsed, 2)
            results.append(result["metrics"])
            _print_metrics(result["metrics"], name, elapsed)
        except Exception as exc:
            logger.error(f"Failed to train {name}: {exc}", exc_info=True)
            print(f"  [ERROR] {name}: {exc}")
    return results


def train_unsupervised(trainer: ModelTrainer) -> list[dict]:
    from src.models.unsupervised import (
        IsolationForestDetector,
        OneClassSVMDetector,
        DBSCANDetector,
        AutoencoderDetector,
    )

    cfg = trainer.cfg
    dbscan_cfg   = cfg.get("unsupervised.dbscan", {}) or {}
    iforest_cfg  = cfg.get("unsupervised.isolation_forest", {}) or {}
    ocsvm_cfg    = cfg.get("unsupervised.one_class_svm", {}) or {}
    ae_cfg       = cfg.get("unsupervised.autoencoder", {}) or {}

    models = [
        ("isolation_forest", lambda: IsolationForestDetector(
            n_estimators=int(iforest_cfg.get("n_estimators", 200)),
            contamination=float(iforest_cfg.get("contamination", 0.05)),
            model_dir=str(MODELS_DIR),
        )),
        ("autoencoder", lambda: AutoencoderDetector(
            encoding_dims=ae_cfg.get("encoding_dims", [64, 32, 16]),
            learning_rate=float(ae_cfg.get("learning_rate", 0.001)),
            batch_size=int(ae_cfg.get("batch_size", 256)),
            epochs=int(ae_cfg.get("epochs", 50)),
            patience=int(ae_cfg.get("patience", 10)),
            anomaly_threshold_percentile=int(ae_cfg.get("anomaly_threshold_percentile", 95)),
            model_dir=str(MODELS_DIR),
        )),
        ("one_class_svm", lambda: OneClassSVMDetector(
            kernel=ocsvm_cfg.get("kernel", "rbf"),
            nu=float(ocsvm_cfg.get("nu", 0.05)),
            gamma=ocsvm_cfg.get("gamma", "scale"),
            model_dir=str(MODELS_DIR),
        )),
        ("dbscan", lambda: DBSCANDetector(
            eps=float(dbscan_cfg.get("eps", 3.6)),
            min_samples=int(dbscan_cfg.get("min_samples", 10)),
            model_dir=str(MODELS_DIR),
        )),
    ]

    results = []
    for name, factory in models:
        logger.info(f"{'='*55}")
        logger.info(f"  Training unsupervised: {name}")
        logger.info(f"{'='*55}")
        t0 = time.perf_counter()
        try:
            model = factory()
            result = trainer.run(model, use_smote=False)
            elapsed = time.perf_counter() - t0
            result["metrics"]["training_time_s"] = round(elapsed, 2)
            results.append(result["metrics"])
            _print_metrics(result["metrics"], name, elapsed)
        except Exception as exc:
            logger.error(f"Failed to train {name}: {exc}", exc_info=True)
            print(f"  [ERROR] {name}: {exc}")
    return results


def plot_comparison(all_results: list[dict], tag: str = "all") -> None:
    """Bar-chart comparing all trained models across key metrics."""
    import pandas as pd

    metric_cols = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc"]
    rows = []
    for r in all_results:
        rows.append({
            "model": r.get("model", "?"),
            **{m: r.get(m, 0.0) for m in metric_cols},
        })
    if not rows:
        return

    df = pd.DataFrame(rows).set_index("model")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df))
    width = 0.15
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for i, col in enumerate(metric_cols):
        ax.bar(x + i * width, df[col], width, label=col, color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Model Comparison – DNS Attack Detection ({_device_info()})", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = REPORTS_DIR / f"model_comparison_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Comparison plot saved: {path}")


def plot_training_times(all_results: list[dict]) -> None:
    """Horizontal bar chart of training times."""
    names = [r.get("model", "?") for r in all_results if "training_time_s" in r]
    times = [r["training_time_s"] for r in all_results if "training_time_s" in r]
    if not names:
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.6)))
    y = np.arange(len(names))
    ax.barh(y, times, color="#42A5F5", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Training time (seconds)", fontsize=11)
    ax.set_title(f"Training Time per Model ({_device_info()})", fontsize=13)
    for i, t in enumerate(times):
        ax.text(t + max(times) * 0.01, i, f"{t:.1f}s", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = REPORTS_DIR / "training_times.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Training time plot saved: {path}")


def save_summary(all_results: list[dict]) -> None:
    summary = []
    for r in all_results:
        summary.append({
            "model": r.get("model"),
            "accuracy": round(r.get("accuracy", 0), 4),
            "f1_weighted": round(r.get("f1_weighted", 0), 4),
            "precision_weighted": round(r.get("precision_weighted", 0), 4),
            "recall_weighted": round(r.get("recall_weighted", 0), 4),
            "roc_auc": round(r.get("roc_auc", 0), 4),
            "training_time_s": r.get("training_time_s", 0),
        })
    path = REPORTS_DIR / "training_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved: {path}")


def _print_metrics(metrics: dict, name: str, elapsed: float) -> None:
    print(f"\n{'='*55}")
    print(f"  {name}  ({elapsed:.1f}s)")
    print(f"{'='*55}")
    for key in ("accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc"):
        print(f"  {key:<30}: {metrics.get(key, 0.0):.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all DNS detection models")
    parser.add_argument("--supervised",   action="store_true", help="Train supervised only")
    parser.add_argument("--unsupervised", action="store_true", help="Train unsupervised only")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    run_sup   = args.supervised or (not args.supervised and not args.unsupervised)
    run_unsup = args.unsupervised or (not args.supervised and not args.unsupervised)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  DNS Attack Detection — Full Training Run")
    print(f"  Device: {_device_info()}")
    print(f"{'='*55}\n")

    trainer = ModelTrainer(config_path=args.config, model_dir=str(MODELS_DIR))

    all_results: list[dict] = []

    if run_sup:
        print("\n>>> Supervised models\n")
        sup_results = train_supervised(trainer)
        all_results.extend(sup_results)
        if len(sup_results) > 1:
            plot_comparison(sup_results, tag="supervised")

    if run_unsup:
        print("\n>>> Unsupervised models\n")
        unsup_results = train_unsupervised(trainer)
        all_results.extend(unsup_results)

    if len(all_results) > 1:
        plot_comparison(all_results, tag="all")

    plot_training_times(all_results)
    save_summary(all_results)

    print(f"\n{'='*55}")
    print("  Training complete!")
    print(f"  Reports: {REPORTS_DIR}/")
    print(f"  Models:  {MODELS_DIR}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
