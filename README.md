# DNS Attack Detection with Machine Learning

> Production-ready Python system for detecting DNS-based network attacks using supervised and unsupervised ML/DL models with real-time packet capture.

---

## Table of Contents
1. [Attack Types Covered](#attack-types-covered)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Training Models](#training-models)
7. [Real-Time Detection](#real-time-detection)
8. [REST API](#rest-api)
9. [Evaluation](#evaluation)
10. [Datasets](#datasets)
11. [Configuration](#configuration)
12. [Running Tests](#running-tests)

---

## Attack Types Covered

| Attack | Description |
|--------|-------------|
| **DNS DDoS** | High-rate flood of DNS queries to exhaust resolver resources |
| **DNS Amplification** | Spoofed ANY/TXT queries for DDoS amplification |
| **DNS Tunneling** | Data exfiltration encoded in high-entropy domain labels |
| **Cache Poisoning** | Injecting malicious records into DNS caches |
| **NXDOMAIN Attack** | Mass non-existent domain queries |
| **Data Exfiltration** | Slow data exfiltration via DNS subdomains |
| **Botnet DNS** | Periodic C&C beaconing via DNS |

---

## Architecture

```
dns-attack-detection-ml/
 │
 ├── src/
 │   ├── data_collection/       # Dataset loaders & synthetic data generator
 │   ├── preprocessing/         # No-leakage cleaner, scaler, encoder
 │   ├── feature_engineering/   # DNS-specific features + rolling window aggregation
 │   ├── models/
 │   │   ├── supervised/        # RandomForest · XGBoost · SVM · MLP · LSTM · Ensemble
 │   │   └── unsupervised/      # IsolationForest · OneClassSVM · DBSCAN · Autoencoder
 │   ├── training/              # ModelTrainer + HyperparameterTuner
 │   ├── evaluation/            # Metrics, confusion matrix, ROC, feature importance
 │   ├── realtime_detection/    # PacketCapture · InferenceEngine · AlertManager · API
 │   └── utils/                 # Logger · ConfigLoader · Helpers
 │
 ├── configs/config.yaml        # All hyper-parameters and settings
 ├── datasets/                  # Place raw CSVs here
 ├── models/                    # Saved model artefacts (auto-created)
 ├── logs/                      # JSON structured logs (auto-created)
 ├── reports/                   # Evaluation plots & JSON reports (auto-created)
 ├── notebooks/                 # Jupyter analysis notebooks
 ├── tests/                     # Pytest unit tests
 └── main.py                    # CLI entry-point
```

### Key design decisions
- **No data leakage** – `DataCleaner` and `DNSScaler` are fitted on the training fold only.
- **SMOTE** – applied inside the training fold only.
- **Abstract `BaseDetector`** – enforces `fit / predict / predict_proba / save / load` on all models.
- **Real-time pipeline** – Scapy capture → `asyncio.Queue` → `InferenceEngine` → `AlertManager`.
- **PyTorch deep learning** – MLP and LSTM with CUDA auto-detection and early stopping.

---

## Project Structure

```
dns-attack-detection-ml/
├── configs/
│   └── config.yaml
├── datasets/            ← place raw CSVs here
├── data/
│   ├── raw/
│   └── processed/
├── logs/                ← JSON logs (auto-created)
├── models/              ← saved artefacts (auto-created)
├── notebooks/
│   └── 01_dns_attack_detection_analysis.ipynb
├── reports/             ← evaluation output (auto-created)
├── src/
│   ├── data_collection/
│   ├── feature_engineering/
│   ├── models/
│   │   ├── supervised/
│   │   └── unsupervised/
│   ├── preprocessing/
│   ├── realtime_detection/
│   ├── training/
│   └── utils/
├── tests/
├── main.py
└── requirements.txt
```

---

## Installation

### Requirements
- Python 3.11+
- (optional) CUDA 11.8+ for GPU-accelerated MLP/LSTM training
- (optional) WinPcap / Npcap (Windows) or libpcap (Linux/macOS) for live capture

> On Windows, prefer `py -3` instead of `python` if `python` points to Python 2.x.

```bash
# 1. Clone the repository
git clone https://github.com/your-org/dns-attack-detection-ml.git
cd dns-attack-detection-ml

# 2. Create a virtual environment
py -3 -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Install dependencies
py -3 -m pip install -r requirements.txt
```

---

## Quick Start

The fastest way to verify the system works end-to-end using the built-in **synthetic dataset**:

```bash
# Train Random Forest on synthetic data (no real dataset needed)
py -3 main.py train --model random_forest

# Train all supervised models
py -3 main.py train --model all

# Train all unsupervised models
py -3 main.py train --model all --unsupervised

# Run tests
py -3 main.py test -v
```

---

## Training Models

### Supervised models
```bash
py -3 main.py train --model random_forest
py -3 main.py train --model xgboost
py -3 main.py train --model svm
py -3 main.py train --model mlp
py -3 main.py train --model lstm
py -3 main.py train --model ensemble_neural   # semi-supervised ensemble (MLP + 2× LSTM variants)
py -3 main.py train --model all          # trains all six supervised models
```

### Unsupervised (anomaly detection)
```bash
py -3 main.py train --model isolation_forest --unsupervised
py -3 main.py train --model one_class_svm    --unsupervised
py -3 main.py train --model dbscan           --unsupervised
py -3 main.py train --model autoencoder      --unsupervised
py -3 main.py train --model all              --unsupervised   # trains all four anomaly detectors
```

### Options
```
--no-smote        Disable SMOTE class balancing
--model-dir DIR   Directory to save model artefacts (default: models/)
--config FILE     Custom YAML config (default: configs/config.yaml)
```

### Using a custom dataset
Place a CSV file under `datasets/` that contains the 29 base features listed in `src/feature_engineering/dns_features.py` plus a `label` column. Then update `configs/config.yaml`:

```yaml
dataset:
  source: "custom"
  custom_path: "datasets/my_dns_dataset.csv"
  label_column: "label"
```

Supported public datasets (set `source` accordingly):
- `cic_dns` – CIC-DNS-2021
- `cira_doh` – CIRA-CIC-DoHBrw-2020
- `unsw_nb15` – UNSW-NB15
- `synthetic` – built-in generator (default, no download required)

---

## Semi-Supervised Ensemble Neural Detector

The **`ensemble_neural`** model combines three neural network architectures with learnable ensemble weighting and optional consistency regularization for semi-supervised learning:

- **MLP** – Feedforward network with batch normalization & dropout (hidden layers: 256→128→64)
- **LSTM v1** – Sequential network (hidden_size=128, num_layers=2) 
- **LSTM v2** – Deeper variant (hidden_size=64, num_layers=3)

### Features
✓ **Learnable ensemble weights** – Network learns optimal combination of the 3 models  
✓ **Consistency regularization** – Optional semi-supervised training on unlabeled benign data  
✓ **DropOut perturbations** – KL divergence between dropout-perturbed and clean predictions  
✓ **Early stopping** – Validation-based model selection  
✓ **CUDA support** – Automatic GPU detection

### Performance
On synthetic DNS data (7,500 test samples):
- **Accuracy:** 99.85%
- **F1-score (weighted):** 99.85%
- **ROC-AUC:** 0.9999958

Per-attack performance: 100% F1 on benign, DNS DDoS, DNS Amplification, Cache Poisoning, NXDOMAIN, Botnet DNS; 99.1% on DNS Tunneling; 98.2% on Data Exfiltration.

### Training
```bash
# Train with default config (50 epochs, batch_size=128)
py -3 main.py train --model ensemble_neural

# Set custom hyperparameters in configs/config.yaml:
#   consistency_lambda: 0.5      # weight of consistency loss
#   dropout_prob: 0.3            # dropout rate in all components
#   learning_rate: 0.001         # Adam learning rate
#   epochs: 50                   # max epochs
#   patience: 10                 # early stopping patience
```

### Configuration (configs/config.yaml)
```yaml
supervised:
  ensemble_neural:
    consistency_lambda: 0.5      # weight for consistency regularization loss
    dropout_prob: 0.3            # dropout in all neural components
    learning_rate: 0.001         # Adam optimizer learning rate
    batch_size: 128              # mini-batch size
    epochs: 50                   # maximum epochs
    patience: 10                 # early stopping patience
```

### Semi-Supervised Learning with Unlabeled Data

The ensemble neural detector supports **consistency regularization** for semi-supervised learning:

```python
from src.models.supervised import SemiSupervisedEnsembleDetector

# Create model with consistency regularization enabled
model = SemiSupervisedEnsembleDetector(
    input_dim=30,
    n_classes=8,
    consistency_lambda=0.5,  # Enable semi-supervised learning
)

# Option 1: Auto-generate synthetic benign unlabeled data
model.fit(X_train, y_train)  # Automatically generates 5000+ unlabeled samples

# Option 2: Use manually-generated unlabeled data
X_unlabeled = model.generate_unlabeled_data(n_samples=10000)
model.fit(X_train, y_train, X_unlabeled=X_unlabeled)

# Option 3: Provide your own unlabeled benign traffic data
model.fit(X_train, y_train, X_unlabeled=X_real_benign)
```

**How it works:**
- KL divergence consistency loss encourages **stable predictions** between dropout-perturbed and clean inputs
- Synthetic benign data is generated with realistic DNS traffic characteristics
- Unlabeled data allows the model to learn better decision boundaries

---

## Hyperparameter Tuning

Optimize ensemble neural detector hyperparameters using **Bayesian optimization** with Optuna:

### Installation
```bash
py -3 -m pip install optuna scikit-optimize
```

### Tuning hyperparameters
```python
from src.training.hyperparameter_tuning_ensemble import EnsembleNeuralHyperparameterTuner

# Create tuner with 50 trials and 5-fold cross-validation
tuner = EnsembleNeuralHyperparameterTuner(
    n_trials=50,
    cv_folds=5,
    scoring="f1_weighted",
)

# Run optimization
results = tuner.tune(X_train, y_train)

# Get best hyperparameters
best_params = results["best_params"]
print(f"Best F1 score: {results['best_value']:.4f}")
print(f"Best hyperparameters: {best_params}")

# Train final model with optimized hyperparameters
model = SemiSupervisedEnsembleDetector(**best_params)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

### Tuned hyperparameters
| Parameter | Range | Description |
|-----------|-------|-------------|
| `consistency_lambda` | [0.0, 1.0] | Weight of consistency regularization loss |
| `dropout_prob` | [0.1, 0.5] | Dropout probability in neural components |
| `learning_rate` | [1e-4, 1e-2] | Adam optimizer learning rate (log-uniform) |
| `batch_size` | {32, 64, 128, 256} | Mini-batch size |
| `patience` | [3, 15] | Early stopping patience (epochs) |

---

## Real-Time Detection

> Requires root / Administrator privileges for raw packet capture.

```bash
# Detect on interface eth0, using saved random_forest model
py -3 main.py detect --interface eth0 --model random_forest

# Set a custom alert threshold and run for 60 s
py -3 main.py detect --interface eth0 --model xgboost --threshold 0.80 --duration 60
```

Windows users: replace `eth0` with the adapter name shown by `ipconfig` (e.g., `Ethernet`).

---

## REST API

The alert microservice exposes a lightweight FastAPI application.

```bash
py -3 main.py api --port 8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/alerts` | List recent alerts (last N, configurable) |
| GET | `/stats` | Alert statistics |
| POST | `/alerts/clear` | Clear the in-memory alert buffer |
| POST | `/alerts/ingest` | Push a pre-computed alert (for testing) |

Interactive docs: http://localhost:8000/docs

---

## Evaluation

After training, evaluation artefacts are saved to `reports/`:

```
reports/
├── random_forest_test_metrics.json          # All scalar metrics
├── random_forest_test_confusion_matrix.png
├── random_forest_test_roc.png
├── random_forest_test_feature_importance.png
└── model_comparison.png                # only when >1 model trained
```

Evaluate a saved model on an external CSV:
```bash
py -3 main.py evaluate --model random_forest --data datasets/test.csv
```

---

## Configuration

All hyper-parameters live in `configs/config.yaml`. You can override any value using environment variables following the pattern `DNS_<SECTION>__<KEY>`:

```bash
export DNS_SUPERVISED__RANDOM_FOREST__N_ESTIMATORS=500
export DNS_LOGGING__LEVEL=DEBUG
py -3 main.py train --model random_forest
```

Key sections:

| Section | Purpose |
|---------|---------|
| `dataset` | Source, paths, train/val/test split ratios |
| `preprocessing` | NaN strategy, IQR outlier clipping |
| `feature_engineering` | Windows sizes, entropy flags |
| `imbalance` | SMOTE k_neighbors, random_state |
| `supervised` | Per-model hyperparameters |
| `unsupervised` | Per-model hyperparameters |
| `realtime` | Capture BPF filter, max queue size, alert threshold |
| `api` | Host, port, alert buffer size, webhook URL |
| `logging` | Level, file path, JSON flag |

---

## Running Tests

```bash
# Via CLI
py -3 main.py test -v

# Run all tests with pytest
py -3 -m pytest tests/ -v --tb=short

# Run only ensemble neural detector tests
py -3 -m pytest tests/test_all.py::TestEnsembleNeuralDetector -v

# Run specific ensemble test
py -3 -m pytest tests/test_all.py::TestEnsembleNeuralDetector::test_consistency_regularization -v

# Run with coverage report
py -3 -m pytest tests/ --cov=src --cov-report=html
```

### Ensemble Neural Detector Tests

Comprehensive test suite (`TestEnsembleNeuralDetector`) with 9 test methods:

| Test | Purpose |
|------|----------|
| `test_model_instantiation` | Verify detector creation with runtime parameter binding |
| `test_fit_predict_basic` | Basic training and inference on synthetic DNS data |
| `test_consistency_regularization` | Consistency loss with explicit unlabeled data |
| `test_auto_generated_unlabeled_data` | Verify auto-generation of synthetic benign samples |
| `test_generate_unlabeled_data` | Manual generation of 1000 unlabeled samples |
| `test_save_and_load` | Model persistence using torch.save/load |
| `test_custom_hyperparameters` | Custom hyperparameter initialization |
| `test_hyperparameter_tuning` | Optuna-based Bayesian optimization (skips if Optuna not installed) |
| `test_ensemble_weights_learning` | Verify learnable ensemble weights are properly initialized |

Run all: `py -3 -m pytest tests/test_all.py::TestEnsembleNeuralDetector -v`

---

## Extending the System

### Adding a new model
1. Create `src/models/<category>/my_model.py` with a class that inherits `BaseDetector`.
2. Implement `fit`, `predict`, `predict_proba` (and optionally `get_params`, `save`, `load`).
   - For PyTorch models: override `save`/`load` to use `torch.save()` (see `LSTMDetector`, `SemiSupervisedEnsembleDetector`)
   - For scikit-learn models: use default `BaseDetector.save/load` with `joblib`
3. Register it in `src/models/<category>/__init__.py`.
4. Add an entry in `main.py`'s `model_map` dictionaries.

### Adding a new attack type
1. Add the label to `configs/config.yaml` under `attack_types`.
2. Add a matching generator method in `src/data_collection/synthetic_generator.py`.

---

## License

MIT – see `LICENSE`.
