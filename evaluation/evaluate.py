"""
evaluate.py
-----------
Consolidated model evaluation report.

Loads saved models + cleaned data to reproduce metrics and build a
summary evaluation table.

Outputs
-------
outputs/evaluation_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
SEED = 42


def _reg_metrics(df: pd.DataFrame) -> dict:
    pkg  = joblib.load(OUTPUT_DIR / "regression_model.pkl")
    feat = pkg["features"]
    scaler = pkg["scaler"]
    model  = pkg["model"]

    X = df[feat].fillna(df[feat].mean())
    y = df["calories_burned"].fillna(df["calories_burned"].mean())
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_test_sc = scaler.transform(X_test)
    y_pred = model.predict(X_test_sc)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":   r2_score(y_test, y_pred),
    }


def _clf_metrics(df: pd.DataFrame) -> dict:
    pkg_rf  = joblib.load(OUTPUT_DIR / "rf_model.pkl")
    pkg_svm = joblib.load(OUTPUT_DIR / "svm_model.pkl")

    avail = pkg_rf["features"]
    X = df[avail].fillna(df[avail].mean())
    y = df["recovery_label_enc"]
    X, y = X.iloc[:-1], y.iloc[:-1]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    rf  = pkg_rf["model"]
    svm = pkg_svm["model"]
    sc  = pkg_svm["scaler"]

    rf_pred   = rf.predict(X_test)
    rf_proba  = rf.predict_proba(X_test)[:, 1]
    svm_pred  = svm.predict(sc.transform(X_test))
    svm_proba = svm.predict_proba(sc.transform(X_test))[:, 1]

    return {
        "RF_F1":      f1_score(y_test, rf_pred,  average="weighted"),
        "RF_AUC":     roc_auc_score(y_test, rf_proba),
        "SVM_F1":     f1_score(y_test, svm_pred, average="weighted"),
        "SVM_AUC":    roc_auc_score(y_test, svm_proba),
    }


def run_evaluation(df: pd.DataFrame) -> None:
    print("[evaluation] Running consolidated evaluation …")
    reg = _reg_metrics(df)
    clf = _clf_metrics(df)

    lines = [
        "=" * 55,
        "  Bimodal Biometric Warehouse – Model Evaluation",
        "=" * 55,
        "",
        "Regression (Calorie Burn Prediction)",
        "-" * 40,
        f"  RMSE           : {reg['RMSE']:.2f} kcal",
        f"  R²             : {reg['R2']:.4f}",
        "",
        "Classification (Next-Day Recovery Prediction)",
        "-" * 40,
        f"  Random Forest  F1 (weighted) : {clf['RF_F1']:.4f}",
        f"  Random Forest  ROC-AUC       : {clf['RF_AUC']:.4f}",
        f"  SVM (RBF)      F1 (weighted) : {clf['SVM_F1']:.4f}",
        f"  SVM (RBF)      ROC-AUC       : {clf['SVM_AUC']:.4f}",
        "",
        "=" * 55,
    ]
    report = "\n".join(lines)
    print(report)

    out = OUTPUT_DIR / "evaluation_summary.txt"
    out.write_text(report)
    print(f"[evaluation] Summary saved → {out}")


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_DIR / "cleaned_data.csv")
    run_evaluation(df)
