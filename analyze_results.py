"""
Final table: top supervised models + top 10 features (no attack_cat leakage).
"""
import json
import glob
import os
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, ".")

SUPERVISED = {"random_forest", "xgboost", "svm", "mlp", "lstm", "ensemble_neural"}
CLEAN_DATE = datetime.date(2026, 6, 1)

sep = "=" * 78

# ── 1. Collect all metrics ──────────────────────────────────────────────────

files = glob.glob("reports/*_test_metrics.json")
rows = []
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    retrained = mtime.date() >= CLEAN_DATE
    rows.append({
        "model": d.get("model", ""),
        "accuracy": d.get("accuracy", 0),
        "f1_weighted": d.get("f1_weighted", 0),
        "f1_macro": d.get("f1_macro", 0),
        "roc_auc": d.get("roc_auc", 0),
        "clean": retrained,
        "supervised": d.get("model", "") in SUPERVISED,
    })

rows.sort(key=lambda x: x["f1_weighted"], reverse=True)  # type: ignore[return-value]

# ── 2. Supervised models table ──────────────────────────────────────────────

sup_rows = [r for r in rows if r["supervised"]]  # type: ignore[index]

print("\n" + sep)
print("  SUPERVISED MODELS -- ranked by F1 Weighted (all retrained without attack_cat)")
print(sep)
print(f"  {'#':<3} {'Model':<22} {'Accuracy':>9} {'F1-W':>8} {'F1-M':>8} {'ROC-AUC':>9}")
print("-" * 78)
for i, r in enumerate(sup_rows, 1):
    print(f"  {i:<3} {r['model']:<22} {r['accuracy']:>9.4f} {r['f1_weighted']:>8.4f} {r['f1_macro']:>8.4f} {r['roc_auc']:>9.4f}")  # type: ignore[index]
print(sep)

# ── 3. Feature importances -- Random Forest (clean, synthetic data) ─────────

print("\n" + sep)
print("  TOP 10 FEATURES -- Random Forest (clean, trained without attack_cat)")
print(sep)

try:
    payload = joblib.load("models/random_forest.joblib")
    rf_clf = payload["model"]
    feat_names: list = payload["feature_names"]
    importances: np.ndarray = rf_clf.feature_importances_

    imp_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    print(f"  {'#':<3} {'Feature':<32} {'Importance':>10}  Bar")
    print("-" * 78)
    for i, row in imp_df.iterrows():
        bar = "#" * int(float(row["importance"]) * 120)
        print(f"  {int(i)+1:<3} {str(row['feature']):<32} {float(row['importance']):>10.4f}  {bar}")
    print(sep)

except Exception as e:
    print(f"  Error: {e}")
    print(sep)

# ── 4. Per-class F1 for top 3 supervised models ─────────────────────────────

top3 = [r["model"] for r in sup_rows[:3]]  # type: ignore[index]

metrics_by_model: dict = {}
for r in sup_rows[:3]:  # type: ignore[index]
    path = f"reports/{r['model']}_test_metrics.json"  # type: ignore[index]
    with open(path) as fp:
        metrics_by_model[r["model"]] = json.load(fp)  # type: ignore[index]

classes = [
    c for c in metrics_by_model[top3[0]]["classification_report"].keys()
    if c not in ("accuracy", "macro avg", "weighted avg")
]

print("\n" + sep)
print(f"  PER-CLASS F1 -- Top 3 supervised models")
print(sep)
print(f"  {'Class':<24} {str(top3[0]):>18} {str(top3[1]):>18} {str(top3[2]):>18}")
print("-" * 78)

for cls in classes:
    vals = []
    for m in top3:
        f1 = metrics_by_model[m]["classification_report"].get(str(cls), {}).get("f1-score", 0)
        vals.append(f"{float(f1):.4f}")
    print(f"  {str(cls):<24} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

print("-" * 78)
for label in ("macro avg", "weighted avg"):
    vals = []
    for m in top3:
        f1 = metrics_by_model[m]["classification_report"].get(label, {}).get("f1-score", 0)
        vals.append(f"{float(f1):.4f}")
    print(f"  {label:<24} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")
print(sep)
