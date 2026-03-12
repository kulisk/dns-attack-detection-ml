"""
DNS Attack Detection ML System
==============================
Main CLI entry-point.

Usage examples
--------------
# Train all supervised models on the synthetic dataset
python main.py train --model all

# Train a specific model
python main.py train --model random_forest

# Train unsupervised models
python main.py train --model isolation_forest --unsupervised

# Run real-time detection (requires root / WinPcap)
python main.py detect --interface eth0 --model random_forest

# Start the FastAPI alert service
python main.py api --port 8000

# Evaluate a saved model on a test CSV
python main.py evaluate --model random_forest --data datasets/test.csv

# Run the test suite
python main.py test
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_logger

logger = get_logger("main", log_file="logs/dns_detection.log")

# ──────────────────────────── CLI ─────────────────────────────


@click.group()
@click.option("--config", default="configs/config.yaml", help="Config file path.")
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """DNS Attack Detection ML System."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


# ──────────────────────────── train ───────────────────────────


@cli.command()
@click.option(
    "--model",
    default="random_forest",
    show_default=True,
    type=click.Choice(
        ["all", "random_forest", "xgboost", "svm", "mlp", "lstm",
         "isolation_forest", "one_class_svm", "dbscan", "autoencoder"],
        case_sensitive=False,
    ),
    help="Model to train.",
)
@click.option("--unsupervised", is_flag=True, default=False, help="Train unsupervised models.")
@click.option("--no-smote", is_flag=True, default=False, help="Disable SMOTE oversampling.")
@click.option("--model-dir", default="models", show_default=True)
@click.pass_context
def train(
    ctx: click.Context,
    model: str,
    unsupervised: bool,
    no_smote: bool,
    model_dir: str,
) -> None:
    """Train one or all detection models."""
    from src.training.trainer import ModelTrainer

    trainer = ModelTrainer(config_path=ctx.obj["config"], model_dir=model_dir)
    use_smote = not no_smote
    unsupervised_models = (
        "isolation_forest", "one_class_svm", "dbscan", "autoencoder"
    )
    supervised_models = (
        "random_forest", "xgboost", "svm", "mlp", "lstm"
    )

    if unsupervised or model in unsupervised_models:
        _train_unsupervised_models(trainer, model if model != "all" else "all", use_smote, model_dir)

    if not unsupervised and model in ("all", *supervised_models):
        _train_supervised_models(trainer, model if model != "all" else "all", use_smote, model_dir)


def _train_supervised_models(trainer, model_name: str, use_smote: bool, model_dir: str) -> None:
    from src.models.supervised import (
        LSTMDetector, MLPDetector, RandomForestDetector, SVMDetector, XGBoostDetector
    )
    model_map = {
        "random_forest": lambda: RandomForestDetector(model_dir=model_dir),
        "xgboost": lambda: XGBoostDetector(model_dir=model_dir),
        "svm": lambda: SVMDetector(model_dir=model_dir),
        "mlp": lambda: MLPDetector(model_dir=model_dir),
        "lstm": lambda: LSTMDetector(model_dir=model_dir),
    }
    keys = list(model_map.keys()) if model_name == "all" else [model_name]
    results = []
    for key in keys:
        if key not in model_map:
            continue
        logger.info(f"=== Training {key} ===")
        model = model_map[key]()
        result = trainer.run(model, use_smote=use_smote)
        results.append(result)
        _print_metrics(result["metrics"], key)

    if len(results) > 1:
        from src.evaluation import Evaluator
        ev = Evaluator(class_names=trainer.label_encoder.classes)
        ev.compare_models([r["metrics"] for r in results])


def _train_unsupervised_models(trainer, model_name: str, use_smote: bool, model_dir: str) -> None:
    from src.models.unsupervised import (
        AutoencoderDetector, DBSCANDetector, IsolationForestDetector, OneClassSVMDetector
    )
    dbscan_cfg = trainer.cfg.get("unsupervised.dbscan", {}) or {}
    model_map = {
        "isolation_forest": lambda: IsolationForestDetector(model_dir=model_dir),
        "one_class_svm": lambda: OneClassSVMDetector(model_dir=model_dir),
        "dbscan": lambda: DBSCANDetector(
            eps=float(dbscan_cfg.get("eps", 0.5)),
            min_samples=int(dbscan_cfg.get("min_samples", 5)),
            algorithm=dbscan_cfg.get("algorithm", "auto"),
            n_jobs=int(dbscan_cfg.get("n_jobs", -1)),
            model_dir=model_dir,
        ),
        "autoencoder": lambda: AutoencoderDetector(model_dir=model_dir),
    }
    keys = list(model_map.keys()) if model_name == "all" else [model_name]
    for key in keys:
        if key not in model_map:
            continue
        logger.info(f"=== Training unsupervised: {key} ===")
        model = model_map[key]()
        result = trainer.run(model, use_smote=False)
        _print_metrics(result["metrics"], key)


# ──────────────────────────── detect ──────────────────────────


@cli.command()
@click.option("--interface", default="eth0", show_default=True, help="Network interface.")
@click.option("--model", default="random_forest", show_default=True, help="Model for inference.")
@click.option("--threshold", default=0.75, show_default=True, type=float)
@click.option("--model-dir", default="models", show_default=True)
@click.option("--duration", default=0, type=float, help="Run for N seconds (0 = indefinitely).")
@click.pass_context
def detect(
    ctx: click.Context,
    interface: str,
    model: str,
    threshold: float,
    model_dir: str,
    duration: float,
) -> None:
    """Start real-time DNS packet capture and inference."""
    import joblib

    from src.models.base_detector import BaseDetector
    from src.realtime_detection.alert_manager import AlertManager
    from src.realtime_detection.inference_engine import InferenceEngine
    from src.realtime_detection.packet_capture import PacketCapture

    # Load model
    detector = _load_model(model, model_dir)

    async def _run() -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        alert_mgr = AlertManager()
        engine = InferenceEngine(
            model=detector,
            packet_queue=queue,
            alert_callback=alert_mgr.handle,
            alert_threshold=threshold,
            model_dir=model_dir,
            config_path=ctx.obj["config"],
        )
        capture = PacketCapture(interface=interface, packet_queue=queue)
        capture.start(loop=asyncio.get_event_loop())
        logger.info("Real-time detection running. Press Ctrl+C to stop.")
        try:
            if duration > 0:
                await engine.run_for(duration)
            else:
                await engine.run()
        finally:
            capture.stop()
            click.echo(f"\nDetection stats: {engine.stats}")

    asyncio.run(_run())


# ──────────────────────────── evaluate ────────────────────────


@cli.command()
@click.option("--model", required=True, help="Model name to evaluate.")
@click.option("--data", required=True, help="Path to test CSV file.")
@click.option("--model-dir", default="models", show_default=True)
@click.pass_context
def evaluate(ctx: click.Context, model: str, data: str, model_dir: str) -> None:
    """Evaluate a saved model on a CSV dataset."""
    import pandas as pd

    from src.evaluation import Evaluator
    from src.training.trainer import ModelTrainer

    detector = _load_model(model, model_dir)
    trainer = ModelTrainer(config_path=ctx.obj["config"], model_dir=model_dir)

    df = pd.read_csv(data)
    X, y = trainer.prepare_features(df)
    ev = Evaluator(class_names=trainer.label_encoder.classes)
    metrics = ev.evaluate(detector, X, y, split="eval")
    _print_metrics(metrics, model)


# ──────────────────────────── api ─────────────────────────────


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, default=False)
def api(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI alert microservice."""
    try:
        import uvicorn
        uvicorn.run("src.realtime_detection.api:app", host=host, port=port, reload=reload)
    except ImportError:
        click.echo("uvicorn is not installed. Run: pip install uvicorn[standard]", err=True)
        sys.exit(1)


# ──────────────────────────── test ────────────────────────────


@cli.command("test")
@click.option("--verbose", "-v", is_flag=True, default=False)
def run_tests(verbose: bool) -> None:
    """Run the project test suite with pytest."""
    import subprocess
    args = ["pytest", "tests/", "--tb=short"]
    if verbose:
        args.append("-v")
    sys.exit(subprocess.call(args))


# ──────────────────────────── Helpers ─────────────────────────


def _load_model(model_name: str, model_dir: str):
    """Load a saved model by name."""
    import joblib
    from pathlib import Path

    from src.models.supervised import (
        LSTMDetector, MLPDetector, RandomForestDetector, SVMDetector, XGBoostDetector
    )
    from src.models.unsupervised import (
        AutoencoderDetector, DBSCANDetector, IsolationForestDetector, OneClassSVMDetector
    )

    model_classes = {
        "random_forest": RandomForestDetector,
        "xgboost": XGBoostDetector,
        "svm": SVMDetector,
        "mlp": MLPDetector,
        "lstm": LSTMDetector,
        "isolation_forest": IsolationForestDetector,
        "one_class_svm": OneClassSVMDetector,
        "dbscan": DBSCANDetector,
        "autoencoder": AutoencoderDetector,
    }

    cls = model_classes.get(model_name)
    if cls is None:
        click.echo(f"Unknown model: {model_name}", err=True)
        sys.exit(1)

    detector = cls(model_dir=model_dir)
    try:
        detector.load()
    except FileNotFoundError:
        click.echo(
            f"No saved model found for '{model_name}'. Train it first with:\n"
            f"  python main.py train --model {model_name}",
            err=True,
        )
        sys.exit(1)
    return detector


def _print_metrics(metrics: dict, model_name: str) -> None:
    click.echo(f"\n{'='*50}")
    click.echo(f"  Results for: {model_name}")
    click.echo(f"{'='*50}")
    for key in ("accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc"):
        val = metrics.get(key, 0.0)
        click.echo(f"  {key:<28}: {val:.4f}")
    click.echo(f"{'='*50}\n")


# ──────────────────────────── Entry ───────────────────────────

if __name__ == "__main__":
    cli()
