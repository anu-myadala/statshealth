"""
flask_app.py
------------
Lightweight Flask REST API that wraps the trained models.

Endpoints
---------
POST /predict/recovery
    Input  (JSON): planned workout and nutritional features for tomorrow
    Output (JSON): predicted recovery state + probability

POST /predict/calories
    Input  (JSON): today's nutritional and activity features
    Output (JSON): predicted calorie expenditure

GET  /health
    Returns service health status.

Usage
-----
    python app/flask_app.py
    # API runs on http://0.0.0.0:5000

Example request (recovery):
    curl -X POST http://localhost:5000/predict/recovery \
         -H "Content-Type: application/json" \
         -d '{"steps": 9000, "active_minutes": 55, "resting_hr": 62,
              "sleep_hours": 7.5, "calories_burned": 2400, "intensity_score": 2,
              "protein_g": 160, "carbs_g": 180, "fat_g": 60,
              "is_high_protein": 1, "is_strength": 1, "is_cardio": 0,
              "rolling_7d_steps": 8000, "rolling_7d_sleep": 7.2,
              "rolling_7d_protein": 145, "prev_intensity": 1}'
"""

import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# ── Load models at startup ────────────────────────────────────────────────────
def _load(name: str):
    path = OUTPUT_DIR / name
    if path.exists():
        return joblib.load(path)
    return None


RF_PKG  = _load("rf_model.pkl")
REG_PKG = _load("regression_model.pkl")

RF_FEATURES  = RF_PKG["features"]  if RF_PKG  else []
REG_FEATURES = REG_PKG["features"] if REG_PKG else []


def _extract(payload: dict, features: list) -> np.ndarray:
    """Extract feature vector from request payload; default missing to 0."""
    return np.array([[float(payload.get(f, 0)) for f in features]])


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "rf_loaded":  RF_PKG  is not None,
        "reg_loaded": REG_PKG is not None,
    })


@app.route("/predict/recovery", methods=["POST"])
def predict_recovery():
    if RF_PKG is None:
        return jsonify({"error": "Random Forest model not loaded. Run main.py first."}), 503

    payload = request.get_json(force=True)
    X       = _extract(payload, RF_FEATURES)
    model   = RF_PKG["model"]
    label_enc = int(model.predict(X)[0])
    proba     = float(model.predict_proba(X)[0][label_enc])
    label     = "Ready to Train" if label_enc == 1 else "Needs Rest"

    return jsonify({
        "recovery_label":       label,
        "recovery_label_enc":   label_enc,
        "confidence":           round(proba, 4),
        "input_features":       {f: payload.get(f, 0) for f in RF_FEATURES},
    })


@app.route("/predict/calories", methods=["POST"])
def predict_calories():
    if REG_PKG is None:
        return jsonify({"error": "Regression model not loaded. Run main.py first."}), 503

    payload  = request.get_json(force=True)
    X        = _extract(payload, REG_FEATURES)
    X_sc     = REG_PKG["scaler"].transform(X)
    calories = float(REG_PKG["model"].predict(X_sc)[0])

    return jsonify({
        "predicted_calories_burned": round(calories, 1),
        "input_features":            {f: payload.get(f, 0) for f in REG_FEATURES},
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
