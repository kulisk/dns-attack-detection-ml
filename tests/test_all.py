"""
Unit tests for the DNS Attack Detection project.

Run with: pytest tests/ -v --tb=short
"""
import asyncio
import math

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

# ─────────────────────────── Helpers ───────────────────────────


def _make_feature_df(n: int = 100, label: str = "benign") -> pd.DataFrame:
    """Create a small feature DataFrame for testing."""
    rng = np.random.default_rng(0)
    cols = [
        "query_length", "entropy", "num_subdomains", "num_labels",
        "max_label_length", "avg_label_length", "digit_ratio", "hyphen_ratio",
        "consonant_ratio", "query_frequency", "nxdomain_ratio", "ttl_mean",
        "ttl_std", "ttl_min", "ttl_max", "packet_size", "packet_size_std",
        "req_resp_ratio", "unique_src_ips", "unique_dst_ips",
        "query_rate_10s", "query_rate_30s", "query_rate_60s",
        "is_any_query", "is_tcp", "response_code",
        "answer_count", "authority_count", "has_valid_tld",
    ]
    data = {c: rng.uniform(0, 1, n) for c in cols}
    data["label"] = label
    return pd.DataFrame(data)


# ─────────────────────────── Utils ─────────────────────────────

class TestHelpers:
    def test_entropy_uniform_string(self):
        from src.utils.helpers import compute_entropy
        s = "abcdefgh"  # 8 unique chars → entropy = log2(8) = 3
        assert abs(compute_entropy(s) - math.log2(8)) < 1e-6

    def test_entropy_empty_string(self):
        from src.utils.helpers import compute_entropy
        assert compute_entropy("") == 0.0

    def test_entropy_constant_string(self):
        from src.utils.helpers import compute_entropy
        assert compute_entropy("aaaa") == 0.0

    def test_extract_domain_features_normal(self):
        from src.utils.helpers import extract_domain_features
        feats = extract_domain_features("sub.example.com")
        assert feats["num_subdomains"] == 1
        assert feats["tld"] == "com"
        assert feats["is_ip"] == 0
        assert feats["query_length"] == len("sub.example.com")

    def test_extract_domain_features_ip(self):
        from src.utils.helpers import extract_domain_features
        feats = extract_domain_features("192.168.1.1")
        assert feats["is_ip"] == 1

    def test_validate_ip_valid(self):
        from src.utils.helpers import validate_ip
        assert validate_ip("8.8.8.8") is True
        assert validate_ip("::1") is True

    def test_validate_ip_invalid(self):
        from src.utils.helpers import validate_ip
        assert validate_ip("not.an.ip") is False


# ─────────────────────────── Synthetic Generator ───────────────


class TestSyntheticGenerator:
    def test_generate_returns_dataframe(self):
        from src.data_collection.synthetic_generator import SyntheticDNSGenerator
        gen = SyntheticDNSGenerator(random_state=1)
        df = gen.generate(n_samples=200)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        assert "label" in df.columns

    def test_generate_all_classes_present(self):
        from src.data_collection.synthetic_generator import ATTACK_TYPES, SyntheticDNSGenerator
        gen = SyntheticDNSGenerator(random_state=2)
        df = gen.generate(n_samples=5000)
        generated_labels = set(df["label"].unique())
        for cls in ATTACK_TYPES:
            assert cls in generated_labels, f"Missing class: {cls}"

    def test_no_missing_features(self):
        from src.data_collection.synthetic_generator import SyntheticDNSGenerator
        gen = SyntheticDNSGenerator(random_state=3)
        df = gen.generate(n_samples=100)
        assert df.isnull().sum().sum() == 0


# ─────────────────────────── DataCleaner ──────────────────────


class TestDataCleaner:
    def test_fit_transform_no_missing(self):
        from src.preprocessing.data_cleaner import DataCleaner
        df = _make_feature_df(100)
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert result.isnull().sum().sum() == 0

    def test_impute_missing_median(self):
        from src.preprocessing.data_cleaner import DataCleaner
        df = _make_feature_df(100)
        df.loc[0, "entropy"] = float("nan")
        cleaner = DataCleaner(missing_strategy="median")
        result = cleaner.fit_transform(df)
        assert not result["entropy"].isna().any()

    def test_infinite_values_replaced(self):
        from src.preprocessing.data_cleaner import DataCleaner
        df = _make_feature_df(50)
        df.loc[0, "ttl_mean"] = float("inf")
        df.loc[1, "ttl_mean"] = float("-inf")
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert np.isfinite(result["ttl_mean"]).all()

    def test_no_leakage_transform_uses_train_stats(self):
        from src.preprocessing.data_cleaner import DataCleaner
        train_df = _make_feature_df(200)
        test_df = _make_feature_df(50)
        test_df.loc[0, "entropy"] = float("nan")
        cleaner = DataCleaner(missing_strategy="median")
        cleaner.fit(train_df)
        result = cleaner.transform(test_df)
        assert not result["entropy"].isna().any()


# ─────────────────────────── DNSScaler ──────────────────────────


class TestDNSScaler:
    def test_fit_transform_standard(self):
        from src.preprocessing.scaler import DNSScaler
        df = _make_feature_df(100)
        feat_cols = [c for c in df.columns if c != "label"]
        scaler = DNSScaler(method="standard", exclude_cols=["label"])
        result = scaler.fit_transform(df)
        # After standard scaling, mean ≈ 0
        assert abs(result[feat_cols].mean().mean()) < 0.5

    def test_no_leakage(self):
        from src.preprocessing.scaler import DNSScaler
        train_df = _make_feature_df(200)
        test_df = _make_feature_df(50)
        scaler = DNSScaler()
        scaler.fit(train_df)
        result = scaler.transform(test_df)
        assert result.shape == test_df.shape


# ─────────────────────────── LabelEncoder ───────────────────────


class TestLabelEncoder:
    def test_encode_decode_roundtrip(self):
        from src.preprocessing.encoder import ATTACK_CLASSES, LabelEncoder
        enc = LabelEncoder(known_classes=ATTACK_CLASSES)
        labels = ["benign", "dns_ddos", "botnet_dns"]
        encoded = enc.transform(labels)
        decoded = enc.inverse_transform(encoded)
        assert list(decoded) == labels

    def test_n_classes(self):
        from src.preprocessing.encoder import ATTACK_CLASSES, LabelEncoder
        enc = LabelEncoder(known_classes=ATTACK_CLASSES)
        assert enc.n_classes == len(ATTACK_CLASSES)


# ─────────────────────────── Feature Extractor ──────────────────


class TestDNSFeatureExtractor:
    def test_transform_adds_columns(self):
        from src.feature_engineering.dns_features import DNSFeatureExtractor
        df = _make_feature_df(50)
        extractor = DNSFeatureExtractor()
        result = extractor.transform(df)
        assert len(result.columns) > len(df.columns)

    def test_extract_from_packet(self):
        from src.feature_engineering.dns_features import DNSFeatureExtractor
        extractor = DNSFeatureExtractor()
        pkt = {
            "qname": "sub.example.com",
            "packet_size": 100,
            "rcode": 0,
            "ttl": 300,
            "proto": "udp",
            "query_frequency": 5,
        }
        features = extractor.extract_from_packet(pkt)
        assert isinstance(features, dict)
        assert "entropy" in features
        assert "query_length" in features


# ─────────────────────────── Window Aggregator ──────────────────


class TestWindowAggregator:
    def test_empty_features(self):
        from src.feature_engineering.window_aggregator import WindowAggregator
        agg = WindowAggregator(windows=[10, 60])
        feats = agg.get_features("10.0.0.1")
        assert feats["query_rate_10s"] == 0.0

    def test_query_rate_accumulates(self):
        import time
        from src.feature_engineering.window_aggregator import WindowAggregator
        agg = WindowAggregator(windows=[60])
        ts = time.time()
        for i in range(10):
            agg.update("10.0.0.1", False, 60, 0, ts=ts + i)
        feats = agg.get_features("10.0.0.1", ts=ts + 9)
        assert feats["query_rate_60s"] > 0.0


# ─────────────────────────── Random Forest ──────────────────────


class TestRandomForestDetector:
    def test_fit_predict(self):
        from src.models.supervised.random_forest import RandomForestDetector
        rng = np.random.default_rng(42)
        X = rng.random((200, 10))
        y = rng.integers(0, 3, 200)
        model = RandomForestDetector(n_estimators=10, model_dir="models")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert model.is_fitted

    def test_predict_proba_shape(self):
        from src.models.supervised.random_forest import RandomForestDetector
        rng = np.random.default_rng(0)
        X = rng.random((100, 10))
        y = rng.integers(0, 3, 100)
        model = RandomForestDetector(n_estimators=5, model_dir="models")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ─────────────────────────── Isolation Forest ───────────────────


class TestIsolationForestDetector:
    def test_fit_predict_binary(self):
        from src.models.unsupervised.isolation_forest import IsolationForestDetector
        rng = np.random.default_rng(0)
        X = rng.random((200, 10))
        model = IsolationForestDetector(n_estimators=20, model_dir="models")
        model.fit(X)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})


# ─────────────────────────── Autoencoder ────────────────────────


class TestAutoencoderDetector:

    # ─────────────────────────── Ensemble Neural Detector ──────────────────────

    class TestEnsembleNeuralDetector:
    class TestEnsembleNeuralDetector:
        """Comprehensive tests for semi-supervised ensemble neural detector."""

        @pytest.mark.timeout(30)
        def test_model_instantiation(self):
            """Test that model can be instantiated with default parameters."""
            from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
        
            model = SemiSupervisedEnsembleDetector(
                input_dim=50,
                n_classes=8,
                model_dir="models",
            )
            assert model.input_dim == 50
            assert model.n_classes == 8
            assert model.consistency_lambda == 0.5
            assert model.dropout_prob == 0.3
            params = model.get_params()
            assert params["input_dim"] == 50
            assert params["n_classes"] == 8

        @pytest.mark.timeout(60)
        def test_fit_predict_basic(self):
            """Test basic fit and predict on synthetic data."""
            from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
            from src.data_collection.synthetic_generator import SyntheticDNSGenerator
        
            # Generate small dataset
            gen = SyntheticDNSGenerator(random_state=42)
            df = gen.generate(n_samples=500)
        
            X = df.drop("label", axis=1).values.astype(np.float32)
            y = np.array([{"benign": 0, "dns_ddos": 1, "dns_amplification": 2,
                          "dns_tunneling": 3, "cache_poisoning": 4, 
                          "nxdomain_attack": 5, "data_exfiltration": 6,
                          "botnet_dns": 7}.get(label, 0) for label in df["label"]])
        
            model = SemiSupervisedEnsembleDetector(
                input_dim=X.shape[1],
                n_classes=len(np.unique(y)),
                epochs=3,
                patience=2,
                consistency_lambda=0.0,  # Disable regularization for speed
                model_dir="models",
            )
        
            # Fit
            model.fit(X[:300], y[:300], X_val=X[300:], y_val=y[300:])
            assert model._is_fitted is True
        
            # Predict
            preds = model.predict(X[:50])
            assert len(preds) == 50
            assert set(preds).issubset(range(len(np.unique(y))))
        
            # Predict proba
            proba = model.predict_proba(X[:50])
            assert proba.shape == (50, len(np.unique(y)))
            assert np.allclose(proba.sum(axis=1), 1.0)

        @pytest.mark.timeout(60)
        def test_consistency_regularization(self):
            """Test consistency regularization with unlabeled data."""
            from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
        
            rng = np.random.default_rng(42)
            X_labeled = rng.random((100, 30)).astype(np.float32)
            y_labeled = rng.integers(0, 4, 100)
            X_unlabeled = rng.random((200, 30)).astype(np.float32)
        
            model = SemiSupervisedEnsembleDetector(
                input_dim=30,
                n_classes=4,
                consistency_lambda=0.5,
                epochs=2,
                patience=1,
                model_dir="models",
            )
        
            # Train with unlabeled data
            model.fit(
                X_labeled,
                y_labeled,
                X_unlabeled=X_unlabeled,
            )
            assert model._is_fitted is True

        @pytest.mark.timeout(60)
        def test_auto_generated_unlabeled_data(self):
            """Test that unlabeled data is auto-generated when not provided."""
            from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
        
            rng = np.random.default_rng(42)
            X = rng.random((100, 30)).astype(np.float32)
            y = rng.integers(0, 4, 100)
        
            model = SemiSupervisedEnsembleDetector(
                input_dim=30,
                n_classes=4,
                consistency_lambda=0.5,  # Will trigger auto-generation
                epochs=2,
                patience=1,
                model_dir="models",
            )
        
            # X_unlabeled is None, should be auto-generated
            model.fit(X, y)
            assert model._is_fitted is True

        @pytest.mark.timeout(30)
        def test_generate_unlabeled_data(self):
            """Test synthetic benign data generation."""
            from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
        
            model = SemiSupervisedEnsembleDetector(input_dim=50, n_classes=8, model_dir="models")
        
            X_benign = model.generate_unlabeled_data(n_samples=1000)
            assert X_benign.shape == (1000, 50)
            assert X_benign.dtype == np.float32
            # Check that values are in reasonable ranges
            assert np.all(X_benign >= 0)
            assert np.all(X_benign <= 100)

        @pytest.mark.timeout(60)
        def test_save_and_load(self):
            """Test model persistence."""
            from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
        
            rng = np.random.default_rng(42)
            X = rng.random((50, 30)).astype(np.float32)
            y = rng.integers(0, 3, 50)
        
            model1 = SemiSupervisedEnsembleDetector(
                input_dim=30,
                n_classes=3,
                learning_rate=0.001,
                epochs=2,
                patience=1,
                model_dir="models",
            )
            model1.fit(X, y)
        
            # Save
            path = model1.save("ensemble_neural_test.pt")
            assert Path(path).exists()
        
            # Load
            model2 = SemiSupervisedEnsembleDetector(model_dir="models")
            model2.load("ensemble_neural_test.pt")
        
            # Predictions should match
            preds1 = model1.predict(X[:10])
            preds2 = model2.predict(X[:10])
            assert np.array_equal(preds1, preds2)
        
            # Clean up
            Path("models/ensemble_neural_test.pt").unlink(missing_ok=True)

        @pytest.mark.timeout(30)
        def test_custom_hyperparameters(self):
            """Test model with custom hyperparameters."""
            from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
        
            rng = np.random.default_rng(42)
            X = rng.random((50, 20)).astype(np.float32)
            y = rng.integers(0, 2, 50)
        
            model = SemiSupervisedEnsembleDetector(
                input_dim=20,
                n_classes=2,
                consistency_lambda=0.8,
                dropout_prob=0.4,
                learning_rate=0.0005,
                batch_size=16,
                epochs=2,
                patience=1,
                model_dir="models",
            )
        
            params = model.get_params()
            assert params["consistency_lambda"] == 0.8
            assert params["dropout_prob"] == 0.4
            assert params["learning_rate"] == 0.0005
            assert params["batch_size"] == 16
        
            model.fit(X, y)
            assert model._is_fitted is True

        @pytest.mark.timeout(60)
        def test_hyperparameter_tuning(self):
            """Test hyperparameter tuning with Optuna."""
            try:
                from src.training.hyperparameter_tuning_ensemble import EnsembleNeuralHyperparameterTuner
            except ImportError:
                pytest.skip("Optuna not installed")
        
            rng = np.random.default_rng(42)
            X = rng.random((200, 30)).astype(np.float32)
            y = rng.integers(0, 4, 200)
        
            tuner = EnsembleNeuralHyperparameterTuner(
                n_trials=3,  # Short test
                cv_folds=2,
                model_dir="models",
            )
        
            results = tuner.tune(X, y)
        
            assert "best_params" in results
            assert "best_value" in results
            assert "study" in results
            assert results["best_value"] > 0  # Should have some score
        
            best_params = results["best_params"]
            assert "consistency_lambda" in best_params
            assert "dropout_prob" in best_params
            assert "learning_rate" in best_params

        @pytest.mark.timeout(30)
        def test_ensemble_weights_learning(self):
            """Test that ensemble weights are learnable and updated during training."""
            from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector, _EnsembleNet
        
            # Create network
            net = _EnsembleNet(input_dim=20, n_classes=3, dropout_rate=0.3)
        
            # Check that ensemble_weights is a parameter
            assert hasattr(net, 'ensemble_weights')
            assert isinstance(net.ensemble_weights, torch.nn.Parameter)
        
            # Verify initial weights
            import torch
            initial_weights = net.ensemble_weights.data.clone()
            assert torch.allclose(initial_weights, torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32))
    def test_fit_predict(self):
        from src.models.unsupervised.autoencoder import AutoencoderDetector
        rng = np.random.default_rng(1)
        X = rng.random((300, 20)).astype(np.float32)
        model = AutoencoderDetector(
            input_dim=20,
            encoding_dims=[16, 8],
            epochs=3,
            batch_size=64,
            model_dir="models",
        )
        model.fit(X)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})
        scores = model.anomaly_scores(X)
        assert scores.shape == (300,)


# ─────────────────────────── Alert Manager ──────────────────────


class TestAlertManager:
    def test_handle_stores_alert(self):
        from src.realtime_detection.alert_manager import AlertManager
        mgr = AlertManager()
        alert = {
            "src_ip": "1.2.3.4",
            "attack_type": "dns_ddos",
            "confidence": 0.95,
            "domain": "evil.com",
            "model": "random_forest",
            "timestamp": 1000.0,
        }
        asyncio.run(mgr.handle(alert))
        assert mgr.total_alerts == 1

    def test_deduplication(self):
        from src.realtime_detection.alert_manager import AlertManager
        mgr = AlertManager(deduplicate_window=60.0)
        alert = {
            "src_ip": "1.2.3.4",
            "attack_type": "dns_ddos",
            "confidence": 0.9,
            "domain": "evil.com",
            "model": "rf",
            "timestamp": 100.0,
        }
        asyncio.run(mgr.handle(alert))
        alert2 = {**alert, "timestamp": 110.0}  # within 60s window
        asyncio.run(mgr.handle(alert2))
        assert mgr.total_alerts == 1  # duplicate suppressed

    def test_clear(self):
        from src.realtime_detection.alert_manager import AlertManager
        mgr = AlertManager()
        alert = {
            "src_ip": "5.5.5.5",
            "attack_type": "botnet_dns",
            "confidence": 0.8,
            "domain": "bot.net",
            "model": "xgb",
            "timestamp": 200.0,
        }
        asyncio.run(mgr.handle(alert))
        mgr.clear()
        assert mgr.total_alerts == 0
