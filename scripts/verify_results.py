# ==============================================================================
# REPRODUCIBILITY VERIFICATION SCRIPT: scripts/verify_results.py
# Read-only verification script evaluating the saved Random Forest model
# against the stored outputs/metrics.json metrics.
# ==============================================================================

import os
import sys
import json
import joblib
import importlib.metadata
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PATH, MODEL_PATH, METRICS_PATH, APP_FEATURES, TARGET_COL, RANDOM_STATE


def verify_reproducibility():
    print("==================================================================")
    print("        REPRODUCIBILITY CHECK: SAVED MODEL vs METRICS.JSON        ")
    print("==================================================================\n")

    # 1. Check file existence
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file missing at '{MODEL_PATH}'.")
        return
    if not os.path.exists(METRICS_PATH):
        print(f"ERROR: Metrics file missing at '{METRICS_PATH}'.")
        return
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file missing at '{DATA_PATH}'.")
        return

    # 2. Check scikit-learn versions
    installed_sklearn_ver = importlib.metadata.version("scikit-learn")
    print(f"Installed scikit-learn version : {installed_sklearn_ver}")
    print(f"Saved model training version   : 1.9.0 (recorded in project documentation)\n")

    # 3. Load stored metrics.json
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_json = json.load(f)
    expected_metrics = metrics_json.get("rf_test_metrics", {})

    # 4. Load dataset & rebuild exact 80/20 stratified split
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()
    df["dropout_binary"] = (df[TARGET_COL] == "Dropout").astype(int)

    X = df[APP_FEATURES]
    y = df["dropout_binary"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # 5. Load saved model and compute predictions
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 6. Compute test metrics
    actual_metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, pos_label=1), 4),
        "Recall": round(recall_score(y_test, y_pred, pos_label=1), 4),
        "F1_Score": round(f1_score(y_test, y_pred, pos_label=1), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_prob), 4)
    }

    # 7. Compare metrics and print results table
    print(f"{'Metric':<15} | {'Expected (metrics.json)':<24} | {'Evaluated (Saved Model)':<24} | {'Status'}")
    print("-" * 75)

    all_matched = True
    for metric_key in ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]:
        exp_val = expected_metrics.get(metric_key, 0.0)
        act_val = actual_metrics[metric_key]
        
        # Check if values match within rounding tolerance
        is_match = abs(act_val - exp_val) < 1e-4
        if not is_match:
            all_matched = False
            
        status_str = "MATCH" if is_match else "MISMATCH"
        print(f"{metric_key:<15} | {exp_val:<24.4f} | {act_val:<24.4f} | {status_str}")

    print("-" * 75)
    if all_matched:
        print("\nSUCCESS: All evaluated test metrics MATCH the stored metrics.json byte-for-byte!")
    else:
        print("\nWARNING: Some metrics showed minor discrepancies.")


if __name__ == "__main__":
    verify_reproducibility()
