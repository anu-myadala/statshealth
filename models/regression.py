"""
regression.py
-------------
Multi-variable Linear Regression to predict active calorie burn.

Features  : protein_g, intensity_score, active_minutes, steps, sleep_hours,
            is_high_protein, carbs_g, fat_g
Target    : calories_burned

Outputs
-------
outputs/regression_results.txt
outputs/plots/regression_actual_vs_pred.png
outputs/plots/regression_residuals.png
outputs/plots/regression_feature_importance.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
PLOTS_DIR  = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED     = 42
FEATURES = [
    "protein_g", "intensity_score", "active_minutes", "steps",
    "sleep_hours", "is_high_protein", "carbs_g", "fat_g",
]
TARGET   = "calories_burned"


def run_regression(df: pd.DataFrame) -> dict:
    X = df[FEATURES].fillna(df[FEATURES].mean())
    y = df[TARGET].fillna(df[TARGET].mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    y_pred  = model.predict(X_test_sc)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    r2      = r2_score(y_test, y_pred)
    cv_r2   = cross_val_score(model, scaler.transform(X_train), y_train, cv=5, scoring="r2").mean()

    metrics = {"rmse": rmse, "r2": r2, "cv_r2": cv_r2}
    print(f"[regression] RMSE  : {rmse:.2f} kcal")
    print(f"[regression] R²    : {r2:.4f}")
    print(f"[regression] CV R² : {cv_r2:.4f}")

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, color="steelblue")
    lim = [min(y_test.min(), y_pred.min()) - 50, max(y_test.max(), y_pred.max()) + 50]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Calories Burned")
    ax.set_ylabel("Predicted Calories Burned")
    ax.set_title(f"Linear Regression: Actual vs Predicted\nRMSE={rmse:.1f}  R²={r2:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "regression_actual_vs_pred.png", dpi=150)
    plt.close()

    # ── Residuals ────────────────────────────────────────────────────────────
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=15, color="darkorange")
    axes[0].axhline(0, linestyle="--", color="grey")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")
    axes[1].hist(residuals, bins=30, color="darkorange", edgecolor="white")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual (kcal)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "regression_residuals.png", dpi=150)
    plt.close()

    # ── Feature coefficients ─────────────────────────────────────────────────
    coef_df = pd.Series(model.coef_, index=FEATURES).sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    coef_df.plot(kind="barh", ax=ax, color=["crimson" if c < 0 else "seagreen" for c in coef_df])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Linear Regression – Standardised Feature Coefficients")
    ax.set_xlabel("Coefficient (standardised)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "regression_feature_importance.png", dpi=150)
    plt.close()
    print("[regression] Saved regression plots.")

    # Save model
    model_path = OUTPUT_DIR / "regression_model.pkl"
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURES}, model_path)
    print(f"[regression] Model saved → {model_path}")

    # Save text report
    report = (
        f"Linear Regression – Calorie Burn Prediction\n"
        f"{'='*45}\n"
        f"Features  : {', '.join(FEATURES)}\n"
        f"Target    : {TARGET}\n"
        f"Test RMSE : {rmse:.2f} kcal\n"
        f"Test R²   : {r2:.4f}\n"
        f"5-fold CV R² : {cv_r2:.4f}\n\n"
        f"Standardised Coefficients:\n{coef_df.to_string()}\n"
    )
    rpt_path = OUTPUT_DIR / "regression_results.txt"
    rpt_path.write_text(report)
    print(f"[regression] Report saved → {rpt_path}")

    return metrics


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_DIR / "cleaned_data.csv")
    run_regression(df)
