"""
classification.py
-----------------
Binary classification: predict next-day recovery state
  "Ready to Train" (1)  vs  "Needs Rest" (0)

Models
------
1. Random Forest
2. Support Vector Machine (RBF kernel)

Outputs
-------
outputs/classification_report.txt
outputs/plots/confusion_matrix_rf.png
outputs/plots/confusion_matrix_svm.png
outputs/plots/roc_curve.png
outputs/plots/rf_feature_importance.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
import joblib

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
PLOTS_DIR  = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED     = 42
FEATURES = [
    "steps", "active_minutes", "resting_hr", "sleep_hours",
    "calories_burned", "intensity_score", "protein_g", "carbs_g",
    "fat_g", "is_high_protein", "is_strength", "is_cardio",
    "rolling_7d_steps", "rolling_7d_sleep", "rolling_7d_protein",
    "prev_intensity",
]
TARGET = "recovery_label_enc"   # 1 = Ready to Train, 0 = Needs Rest


def _plot_confusion_matrix(y_true, y_pred, title: str, fname: str) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Needs Rest", "Ready to Train"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close()
    print(f"[classification] Saved {fname}")


def _plot_roc(models_data: list) -> None:
    """models_data: list of (name, fpr, tpr, auc)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, fpr, tpr, auc in models_data:
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Recovery State Classification")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close()
    print("[classification] Saved roc_curve.png")


def run_classification(df: pd.DataFrame) -> dict:
    avail = [f for f in FEATURES if f in df.columns]
    X     = df[avail].fillna(df[avail].mean())
    y     = df[TARGET]

    # Drop last row (no next-day recovery info)
    X, y = X.iloc[:-1], y.iloc[:-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)   # RF doesn't need scaling but we use raw features
    rf_pred   = rf.predict(X_test)
    rf_proba  = rf.predict_proba(X_test)[:, 1]
    rf_f1     = f1_score(y_test, rf_pred, average="weighted")
    rf_auc    = roc_auc_score(y_test, rf_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)

    print(f"\n[classification] Random Forest:")
    print(f"  F1 (weighted) : {rf_f1:.4f}")
    print(f"  ROC-AUC       : {rf_auc:.4f}")
    print(classification_report(y_test, rf_pred, target_names=["Needs Rest", "Ready to Train"]))

    _plot_confusion_matrix(y_test, rf_pred, "Random Forest – Confusion Matrix", "confusion_matrix_rf.png")

    # Feature importance
    imp_df = pd.Series(rf.feature_importances_, index=avail).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp_df.tail(15).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Random Forest – Feature Importances (Top 15)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rf_feature_importance.png", dpi=150)
    plt.close()
    print("[classification] Saved rf_feature_importance.png")

    # ── SVM ──────────────────────────────────────────────────────────────────
    svm = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=SEED)
    svm.fit(X_train_sc, y_train)
    svm_pred  = svm.predict(X_test_sc)
    svm_proba = svm.predict_proba(X_test_sc)[:, 1]
    svm_f1    = f1_score(y_test, svm_pred, average="weighted")
    svm_auc   = roc_auc_score(y_test, svm_proba)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_proba)

    print(f"\n[classification] SVM (RBF):")
    print(f"  F1 (weighted) : {svm_f1:.4f}")
    print(f"  ROC-AUC       : {svm_auc:.4f}")
    print(classification_report(y_test, svm_pred, target_names=["Needs Rest", "Ready to Train"]))

    _plot_confusion_matrix(y_test, svm_pred, "SVM (RBF) – Confusion Matrix", "confusion_matrix_svm.png")

    # ── Combined ROC curve ────────────────────────────────────────────────────
    _plot_roc([
        ("Random Forest", rf_fpr, rf_tpr, rf_auc),
        ("SVM",           svm_fpr, svm_tpr, svm_auc),
    ])

    # Save models
    joblib.dump({"model": rf,  "features": avail},                      OUTPUT_DIR / "rf_model.pkl")
    joblib.dump({"model": svm, "scaler": scaler, "features": avail},    OUTPUT_DIR / "svm_model.pkl")
    print("[classification] Models saved to outputs/")

    # Save text report
    report = (
        "Classification Report – Next-Day Recovery Prediction\n"
        "=====================================================\n"
        f"Target   : {TARGET}  (1=Ready to Train, 0=Needs Rest)\n"
        f"Features : {', '.join(avail)}\n\n"
        "--- Random Forest ---\n"
        + classification_report(y_test, rf_pred, target_names=["Needs Rest", "Ready to Train"])
        + f"F1 (weighted): {rf_f1:.4f}  |  ROC-AUC: {rf_auc:.4f}\n\n"
        "--- SVM (RBF) ---\n"
        + classification_report(y_test, svm_pred, target_names=["Needs Rest", "Ready to Train"])
        + f"F1 (weighted): {svm_f1:.4f}  |  ROC-AUC: {svm_auc:.4f}\n"
    )
    rpt = OUTPUT_DIR / "classification_report.txt"
    rpt.write_text(report)
    print(f"[classification] Report saved → {rpt}")

    return {
        "rf_f1": rf_f1, "rf_auc": rf_auc,
        "svm_f1": svm_f1, "svm_auc": svm_auc,
    }


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_DIR / "cleaned_data.csv")
    run_classification(df)
