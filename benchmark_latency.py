"""
Inference latency benchmark for all trained supervised models.

Loads the saved preprocessors and test data, then times each model's
predict() call — both single-sample latency and full-batch throughput.
Updates each model's *_test_metrics.json with the latency fields.
"""
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

sys.path.insert(0, ".")

from src.data_collection import DatasetLoader
from src.feature_engineering import DNSFeatureExtractor
from src.utils.config_loader import ConfigLoader

# ── Model registry ──────────────────────────────────────────────────────────

def load_sklearn_model(name: str):
    """Load a joblib-persisted sklearn/xgboost model."""
    payload = joblib.load(f"models/{name}.joblib")
    return payload["model"]


def load_lstm():
    from src.models.supervised.lstm_model import LSTMDetector
    m = LSTMDetector()
    m.load()
    return m


def load_mlp():
    from src.models.supervised.neural_network import MLPDetector
    m = MLPDetector()
    m.load()
    return m


def load_ensemble():
    from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
    m = SemiSupervisedEnsembleDetector()
    m.load()
    return m


MODELS = {
    "random_forest":  lambda: load_sklearn_model("random_forest"),
    "svm":            lambda: load_sklearn_model("svm"),
    "xgboost":        lambda: load_sklearn_model("xgboost"),
    "mlp":            load_mlp,
    "lstm":           load_lstm,
    "ensemble_neural": load_ensemble,
}

WARMUP_RUNS = 50
TIMED_RUNS  = 100

sep = "=" * 78


# ── Data preparation ─────────────────────────────────────────────────────────

def build_test_features() -> np.ndarray:
    """Reconstruct the scaled test feature matrix using saved preprocessors."""
    cfg = ConfigLoader("configs/config.yaml")

    loader = DatasetLoader(
        dataset_dir=cfg.get("paths.datasets", "datasets"),
        source=cfg.get("dataset.source", "synthetic"),
        test_size=float(cfg.get("dataset.test_size", 0.15)),
        val_size=float(cfg.get("dataset.val_size", 0.15)),
        random_state=int(cfg.get("dataset.random_state", 42)),
        label_col=cfg.get("label_column", "label"),
    )
    df = loader.load()
    _, _, test_df = loader.split(df)

    extractor = DNSFeatureExtractor()
    test_df = extractor.transform(test_df)

    cleaner      = joblib.load("models/cleaner.joblib")
    scaler       = joblib.load("models/scaler.joblib")
    cat_maps     = joblib.load("models/categorical_maps.joblib")

    NON_FEATURE = {"label", "is_attack", "attack_cat"}
    label_col = cfg.get("label_column", "label")

    test_df = cleaner.transform(test_df, label_col)

    feature_cols = [c for c in test_df.columns if c not in NON_FEATURE]

    for col in feature_cols:
        if col in cat_maps:
            mapping = cat_maps[col]
            test_df[col] = (
                test_df[col].astype(str).fillna("__nan__")
                .map(mapping).fillna(-1).astype(float)
            )

    X_test = scaler.transform(test_df[feature_cols]).values
    return X_test


# ── Timing helper ─────────────────────────────────────────────────────────────

def time_model(predict_fn, X: np.ndarray) -> dict:
    """Return latency metrics for a callable predict_fn."""
    # Full-batch timing
    t0 = time.perf_counter()
    predict_fn(X)
    batch_ms = (time.perf_counter() - t0) * 1000

    # Single-sample latency
    x1 = X[:1]
    for _ in range(WARMUP_RUNS):
        predict_fn(x1)

    t0 = time.perf_counter()
    for _ in range(TIMED_RUNS):
        predict_fn(x1)
    single_ms = (time.perf_counter() - t0) / TIMED_RUNS * 1000

    return {
        "inference_latency_ms_single": round(single_ms, 4),
        "inference_latency_ms_batch":  round(batch_ms, 2),
        "throughput_samples_per_sec":  round(len(X) / (batch_ms / 1000), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + sep)
    print("  INFERENCE LATENCY BENCHMARK")
    print(sep)
    print("  Building test feature matrix from saved preprocessors …")
    X_test = build_test_features()
    n_samples = len(X_test)
    print(f"  Test set: {n_samples:,} samples  x  {X_test.shape[1]} features")
    print(sep)

    print(f"\n  {'Model':<22} {'Single (ms)':>12} {'Batch (ms)':>11} {'Throughput (s/s)':>18}")
    print("-" * 70)

    results: dict[str, dict] = {}

    for name, loader_fn in MODELS.items():
        try:
            model = loader_fn()
            # Unified predict callable
            if hasattr(model, "predict"):
                predict_fn = model.predict
            else:
                predict_fn = lambda x: model.predict(x)  # noqa: E731

            latency = time_model(predict_fn, X_test)
            results[name] = latency

            print(
                f"  {name:<22} "
                f"{latency['inference_latency_ms_single']:>12.4f} "
                f"{latency['inference_latency_ms_batch']:>11.2f} "
                f"{latency['throughput_samples_per_sec']:>18,.1f}"
            )

            # Update the metrics JSON
            metrics_path = f"reports/{name}_test_metrics.json"
            try:
                with open(metrics_path) as fp:
                    m = json.load(fp)
                m.update(latency)
                with open(metrics_path, "w") as fp:
                    json.dump(m, fp, indent=2)
            except FileNotFoundError:
                print(f"    [warn] {metrics_path} not found — skipping JSON update")

        except Exception as exc:
            print(f"  {name:<22} ERROR: {exc}")

    print("-" * 70)
    print(sep)
    print("\n  Latency fields written to reports/*_test_metrics.json\n")


if __name__ == "__main__":
    main()
