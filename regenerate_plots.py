"""
Regenerate all evaluation plots for every trained model without retraining.

Loads each saved model, rebuilds the test set via the saved preprocessors,
and runs the Evaluator to produce confusion matrices, ROC curves, and
feature importance plots.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np

sys.path.insert(0, ".")

from src.data_collection import DatasetLoader
from src.feature_engineering import DNSFeatureExtractor
from src.evaluation import Evaluator
from src.utils.config_loader import ConfigLoader

# ── Data preparation ──────────────────────────────────────────────────────────

def build_test_set() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_test, y_test, class_names) using saved preprocessors."""
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
    label_encoder = joblib.load("models/label_encoder.joblib")
    cat_maps     = joblib.load("models/categorical_maps.joblib")

    NON_FEATURE = {"label", "is_attack", "attack_cat"}
    label_col = cfg.get("label_column", "label")

    test_df = cleaner.transform(test_df, label_col)
    y_test  = label_encoder.transform(test_df[label_col])

    feature_cols = [c for c in test_df.columns if c not in NON_FEATURE]

    for col in feature_cols:
        if col in cat_maps:
            mapping = cat_maps[col]
            test_df[col] = (
                test_df[col].astype(str).fillna("__nan__")
                .map(mapping).fillna(-1).astype(float)
            )

    X_test = scaler.transform(test_df[feature_cols]).values
    return X_test, y_test, label_encoder.classes


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_random_forest():
    from src.models.supervised.random_forest import RandomForestDetector
    m = RandomForestDetector()
    return m.load()

def load_xgboost():
    from src.models.supervised.xgboost_model import XGBoostDetector
    m = XGBoostDetector()
    return m.load()

def load_svm():
    from src.models.supervised.svm_model import SVMDetector
    m = SVMDetector()
    return m.load()

def load_mlp():
    from src.models.supervised.neural_network import MLPDetector
    m = MLPDetector()
    return m.load()

def load_lstm():
    from src.models.supervised.lstm_model import LSTMDetector
    m = LSTMDetector()
    return m.load()

def load_ensemble():
    from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
    m = SemiSupervisedEnsembleDetector()
    return m.load()


MODELS = [
    ("random_forest",  load_random_forest),
    ("xgboost",        load_xgboost),
    ("svm",            load_svm),
    ("mlp",            load_mlp),
    ("lstm",           load_lstm),
    ("ensemble_neural", load_ensemble),
]

sep = "=" * 60

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + sep)
    print("  REGENERATING ALL MODEL PLOTS")
    print(sep)
    print("  Building test set …")
    X_test, y_test, class_names = build_test_set()
    print(f"  Test set: {len(X_test):,} samples  x  {X_test.shape[1]} features")
    print(f"  Classes : {class_names}")
    print(sep)

    evaluator = Evaluator(
        class_names=class_names,
        output_dir="reports",
        dpi=150,
    )

    for name, loader_fn in MODELS:
        print(f"\n  [{name}] Loading …", end=" ", flush=True)
        try:
            model = loader_fn()
            print("OK — evaluating …", end=" ", flush=True)
            metrics = evaluator.evaluate(model, X_test, y_test, split="test")
            print(
                f"done  |  F1-W={metrics['f1_weighted']:.4f}  "
                f"lat={metrics.get('inference_latency_ms_single', 0):.3f}ms"
            )
        except Exception as exc:
            print(f"ERROR: {exc}")

    print("\n" + sep)
    print("  All plots saved to reports/")
    print(sep + "\n")


if __name__ == "__main__":
    main()
