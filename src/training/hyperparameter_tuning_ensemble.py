"""
Hyperparameter tuning for semi-supervised ensemble neural detector using Optuna.

This module provides Bayesian optimization for finding optimal hyperparameters
for the EnsembleNeuralDetector model.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.models.supervised.ensemble_neural_detector import SemiSupervisedEnsembleDetector
from src.utils import get_logger

logger = get_logger(__name__)


class EnsembleNeuralHyperparameterTuner:
    """Bayesian hyperparameter optimizer for ensemble neural detector."""

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
        device: Optional[str] = None,
        model_dir: str = "models",
    ) -> None:
        """Initialize the tuner.

        Args:
            n_trials: Number of optimization trials.
            cv_folds: Number of cross-validation folds.
            scoring: Sklearn scoring metric.
            device: Compute device ("cuda" or "cpu").
            model_dir: Directory for model persistence.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")

        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.device = device
        self.model_dir = model_dir

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        """Run hyperparameter tuning using Optuna.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dictionary with best parameters and trial results.
        """
        logger.info(
            "Starting hyperparameter tuning",
            extra={
                "n_trials": self.n_trials,
                "cv_folds": self.cv_folds,
                "n_samples": len(X_train),
            },
        )

        def objective(trial: optuna.Trial) -> float:
            """Objective function for optimization."""
            # Suggest hyperparameters
            consistency_lambda = trial.suggest_float("consistency_lambda", 0.0, 1.0)
            dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.5)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            patience = trial.suggest_int("patience", 3, 15)

            # Create model with suggested hyperparameters
            model = SemiSupervisedEnsembleDetector(
                input_dim=X_train.shape[1],
                n_classes=len(np.unique(y_train)),
                consistency_lambda=consistency_lambda,
                dropout_prob=dropout_prob,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=30,  # Fixed for tuning
                patience=patience,
                device=self.device,
                model_dir=self.model_dir,
            )

            # Cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
                X_train_fold = X_train[train_idx]
                y_train_fold = y_train[train_idx]
                X_test_fold = X_train[test_idx]
                y_test_fold = y_train[test_idx]

                # Train fold
                model.fit(X_train_fold, y_train_fold)

                # Evaluate fold
                y_pred = model.predict(X_test_fold)
                score = _compute_score(y_test_fold, y_pred, self.scoring)
                scores.append(score)

                logger.debug(
                    f"Trial {trial.number}, Fold {fold_idx + 1}/{self.cv_folds}",
                    extra={"score": round(score, 4)},
                )

                # Pruning: stop trial if performance is poor
                trial.report(score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = np.mean(scores)
            logger.info(
                f"Trial {trial.number} completed",
                extra={
                    "mean_score": round(mean_score, 4),
                    "std_score": round(np.std(scores), 4),
                    "params": {
                        "consistency_lambda": round(consistency_lambda, 4),
                        "dropout_prob": round(dropout_prob, 4),
                        "learning_rate": round(learning_rate, 4),
                        "batch_size": batch_size,
                    },
                },
            )

            return mean_score

        # Create optimizer
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # Run optimization
        study.optimize(objective, n_trials=self.n_trials)

        # Get results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        logger.info(
            "Hyperparameter tuning completed",
            extra={
                "best_value": round(best_value, 4),
                "best_params": {k: round(v, 4) if isinstance(v, float) else v 
                               for k, v in best_params.items()},
            },
        )

        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": study,
            "trials_df": study.trials_dataframe(),
        }


def _compute_score(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    """Compute scoring metric."""
    if scoring == "f1_weighted":
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)
    elif scoring == "accuracy":
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    elif scoring == "f1_macro":
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")
