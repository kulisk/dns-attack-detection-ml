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
 │   │   ├── supervised/        # RandomForest · XGBoost · SVM · MLP · LSTM
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

```bash
# 1. Clone the repository
git clone https://github.com/your-org/dns-attack-detection-ml.git
cd dns-attack-detection-ml

# 2. Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

The fastest way to verify the system works end-to-end using the built-in **synthetic dataset**:

```bash
# Train Random Forest on synthetic data (no real dataset needed)
python main.py train --model random_forest

# Train all supervised models
python main.py train --model all

# Run tests
python main.py test -v
```

---

## Training Models

### Supervised models
```bash
python main.py train --model random_forest
python main.py train --model xgboost
python main.py train --model svm
python main.py train --model mlp
python main.py train --model lstm
python main.py train --model all          # trains all five
```

### Unsupervised (anomaly detection)
```bash
python main.py train --model isolation_forest --unsupervised
python main.py train --model one_class_svm    --unsupervised
python main.py train --model dbscan           --unsupervised
python main.py train --model autoencoder      --unsupervised
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

## Real-Time Detection

> Requires root / Administrator privileges for raw packet capture.

```bash
# Detect on interface eth0, using saved random_forest model
python main.py detect --interface eth0 --model random_forest

# Set a custom alert threshold and run for 60 s
python main.py detect --interface eth0 --model xgboost --threshold 0.80 --duration 60
```

Windows users: replace `eth0` with the adapter name shown by `ipconfig` (e.g., `Ethernet`).

---

## REST API

The alert microservice exposes a lightweight FastAPI application.

```bash
python main.py api --port 8000
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
├── random_forest_metrics.json          # All scalar metrics
├── random_forest_confusion_matrix.png
├── random_forest_roc_curves.png
├── random_forest_feature_importance.png
└── model_comparison.png                # only when >1 model trained
```

Evaluate a saved model on an external CSV:
```bash
python main.py evaluate --model random_forest --data datasets/test.csv
```

---

## Configuration

All hyper-parameters live in `configs/config.yaml`. You can override any value using environment variables following the pattern `DNS_<SECTION>__<KEY>`:

```bash
export DNS_SUPERVISED__RANDOM_FOREST__N_ESTIMATORS=500
export DNS_LOGGING__LEVEL=DEBUG
python main.py train --model random_forest
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
python main.py test -v

# Directly with pytest
pytest tests/ -v --tb=short
```

---

## Extending the System

### Adding a new model
1. Create `src/models/<category>/my_model.py` with a class that inherits `BaseDetector`.
2. Implement `fit`, `predict`, `predict_proba` (and optionally `get_params`).
3. Register it in `src/models/<category>/__init__.py`.
4. Add an entry in `main.py`'s `model_map` dictionaries.

### Adding a new attack type
1. Add the label to `configs/config.yaml` under `attack_types`.
2. Add a matching generator method in `src/data_collection/synthetic_generator.py`.

---

## License

MIT – see `LICENSE`.
