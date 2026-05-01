"""
flask_api.py
Lightweight REST API wrapping the trained Random Forest recovery classifier
and the Linear Regression calorie predictor.

Run:
    python app/flask_api.py

Endpoints
---------
GET  /health          — liveness check
POST /predict         — recovery + calorie prediction

Example request body:
{
  "resting_heart_rate": 62,
  "hrv_score": 58,
  "sleep_duration_hours": 7.2,
  "total_active_minutes": 45,
  "active_calories": 350,
  "steps": 9500,
  "protein_g": 180,
  "carbs_g": 210,
  "fat_g": 65,
  "intensity": 7,
  "duration_minutes": 50,
  "is_high_protein": 1,
  "is_poultry": 1,
  "is_vegetarian": 0,
  "is_weekend": 0,
  "PC1": 0.0,
  "PC2": 0.0,
  "PC3": 0.0
}
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

app = Flask(__name__)

# ── load artefacts ────────────────────────────────────────────────────────────
_rf  = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
_lr  = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
_pca_scaler = joblib.load(os.path.join(MODELS_DIR, "pca_scaler.pkl"))
_pca        = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))

with open(os.path.join(MODELS_DIR, "clf_features.json")) as fh:
    CLF_FEATURES = json.load(fh)
with open(os.path.join(MODELS_DIR, "reg_features.json")) as fh:
    REG_FEATURES = json.load(fh)

HR_PCA_FEATURES = [
    "resting_heart_rate", "hrv_score", "total_active_minutes",
    "steps", "active_calories", "duration_minutes", "intensity",
]


def _compute_pcs(data: dict) -> dict:
    """Derive PC1/PC2/PC3 from raw HR features if not already provided."""
    if all(f"PC{i}" in data for i in (1, 2, 3)):
        return data
    vec = np.array([[data.get(f, 0.0) for f in HR_PCA_FEATURES]])
    vec_sc = _pca_scaler.transform(vec)
    pcs = _pca.transform(vec_sc)[0]
    return {**data, "PC1": float(pcs[0]), "PC2": float(pcs[1]), "PC3": float(pcs[2])}


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForest + LinearRegression"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Empty request body"}), 400

    # Derive PCA components if missing
    try:
        payload = _compute_pcs(payload)
    except Exception as exc:
        return jsonify({"error": f"PCA computation failed: {exc}"}), 422

    # ── Recovery classification ───────────────────────────────────────────────
    missing_clf = [f for f in CLF_FEATURES if f not in payload]
    if missing_clf:
        return jsonify({"error": f"Missing classifier features: {missing_clf}"}), 422

    X_clf = np.array([[payload[f] for f in CLF_FEATURES]])
    label = int(_rf.predict(X_clf)[0])
    proba = float(_rf.predict_proba(X_clf)[0][1])

    # ── Calorie regression ────────────────────────────────────────────────────
    missing_reg = [f for f in REG_FEATURES if f not in payload]
    pred_calories = None
    if not missing_reg:
        X_reg = np.array([[payload[f] for f in REG_FEATURES]])
        pred_calories = float(_lr.predict(X_reg)[0])

    return jsonify({
        "recovery_label":       label,
        "recovery_probability": round(proba, 4),
        "message":              "Ready to Train" if label == 1 else "Needs Rest",
        "predicted_active_calories": round(pred_calories, 1) if pred_calories else None,
    })


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"✓ Biometric API listening on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
