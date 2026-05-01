"""
flask_api.py
============
Lightweight REST API wrapping the trained sklearn Pipelines.

Each Pipeline was serialised with `joblib.dump(pipeline, ...)` where the
Pipeline bundles a fitted StandardScaler AND the fitted model.  Calling
`pipeline.predict(raw_input)` therefore always scales correctly — there is
no risk of predicting on un-scaled data.

Run:
    python app/flask_api.py

Endpoints
---------
GET  /health          — liveness check
POST /predict         — recovery classification + calorie prediction

Minimal request body (raw, unscaled values):
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
  "lag1_sleep": 7.5,
  "lag1_active_calories": 400,
  "lag1_hrv": 55
}

If the optional ``minute_level_hr`` array (1440 floats) is provided, the API
derives hr_pc1–hr_pc5 automatically using the pickled HR PCA pipeline.
Otherwise hr_pc1–hr_pc5 default to 0.
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify

# ── paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

app = Flask(__name__)

# ── load full pipelines (scaler + model bundled) ──────────────────────────────
_rf_pipeline  = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.pkl"))
_svm_pipeline = joblib.load(os.path.join(MODELS_DIR, "svm_pipeline.pkl"))
_reg_pipeline = joblib.load(os.path.join(MODELS_DIR, "reg_pipeline.pkl"))
_hr_pca_pipe  = joblib.load(os.path.join(MODELS_DIR, "hr_pca_pipeline.pkl"))

with open(os.path.join(MODELS_DIR, "clf_features.json")) as fh:
    CLF_FEATURES = json.load(fh)
with open(os.path.join(MODELS_DIR, "reg_features.json")) as fh:
    REG_FEATURES = json.load(fh)

HR_PC_NAMES = ["hr_pc1", "hr_pc2", "hr_pc3", "hr_pc4", "hr_pc5"]


def _derive_hr_pcs(payload: dict) -> dict:
    """Compute hr_pc1..hr_pc5 from a 1440-element HR array if provided.
    The HR PCA pipeline (scaler + PCA) is applied exactly as during training.
    If the minute-level array is absent, PCs default to 0 (neutral).
    """
    if "minute_level_hr" in payload:
        hr_arr = np.array(payload["minute_level_hr"], dtype=float)
        if hr_arr.shape != (1440,):
            raise ValueError("minute_level_hr must have exactly 1440 elements")
        pcs = _hr_pca_pipe.transform(hr_arr.reshape(1, -1))[0]
        return {**payload,
                **{f"hr_pc{i+1}": float(pcs[i]) for i in range(5)}}
    # Default: all PCs = 0 (will still work; prediction quality degrades)
    return {**payload, **{pc: payload.get(pc, 0.0) for pc in HR_PC_NAMES}}


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "classifiers": ["rf_pipeline", "svm_pipeline"],
        "regressor":   "reg_pipeline",
        "note": "Pipelines contain fitted StandardScaler + model; input raw values.",
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Empty request body"}), 400

    # Derive HR PCA scores from minute-level array (or use defaults)
    try:
        payload = _derive_hr_pcs(payload)
    except (ValueError, Exception) as exc:
        return jsonify({"error": f"HR PCA derivation failed: {exc}"}), 422

    # ── Recovery classification (RF pipeline) ─────────────────────────────────
    missing_clf = [f for f in CLF_FEATURES if f not in payload]
    if missing_clf:
        return jsonify({"error": f"Missing classifier features: {missing_clf}"}), 422

    # Build raw feature vector — Pipeline handles scaling internally
    X_clf = np.array([[payload[f] for f in CLF_FEATURES]])
    label     = int(_rf_pipeline.predict(X_clf)[0])
    proba_rf  = float(_rf_pipeline.predict_proba(X_clf)[0][1])
    proba_svm = float(_svm_pipeline.predict_proba(X_clf)[0][1])

    # ── Calorie regression (reg pipeline) ────────────────────────────────────
    missing_reg = [f for f in REG_FEATURES if f not in payload]
    pred_calories = None
    if not missing_reg:
        X_reg = np.array([[payload[f] for f in REG_FEATURES]])
        pred_calories = float(_reg_pipeline.predict(X_reg)[0])

    return jsonify({
        "recovery_label":             label,
        "recovery_probability_rf":    round(proba_rf,  4),
        "recovery_probability_svm":   round(proba_svm, 4),
        "message":                    "Ready to Train" if label == 1 else "Needs Rest",
        "predicted_active_calories":  round(pred_calories, 1) if pred_calories is not None else None,
        "note": "Input was raw (unscaled). Scaling applied inside each Pipeline.",
    })


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"✓ Biometric API listening on http://0.0.0.0:{port}")
    print(f"  Classifiers: StandardScaler + RandomForest | StandardScaler + SVM")
    print(f"  Regressor  : StandardScaler + LinearRegression")
    app.run(host="0.0.0.0", port=port, debug=False)
