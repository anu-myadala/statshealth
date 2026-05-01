# The Bimodal Biometric Warehouse
### Mining Wearable and Nutritional Data for Metabolic Optimization

A full end-to-end data-science project that predicts **physiological recovery**
and **caloric expenditure** for "bimodal" activity profiles — days of intense
exercise (heavy dumbbell training or high-incline walking) followed by highly
sedentary recovery periods.

---

## Project Structure

```
statshealth/
├── data/
│   ├── generate_data.py          # Synthetic wearable + nutrition data (365 days)
│   ├── dim_workout.csv
│   ├── dim_nutrition.csv
│   ├── dim_time.csv
│   └── fact_daily_biometrics.csv
├── database/
│   ├── schema.sql                # Star-schema DDL (SQLite / MySQL / PostgreSQL)
│   ├── load_data.py              # Loads CSVs into SQLite
│   └── biometric_warehouse.db   # Auto-generated SQLite database
├── notebooks/
│   └── bimodal_biometric_warehouse.ipynb   # Main analysis notebook
├── app/
│   ├── flask_api.py              # REST API for the trained models
│   └── models/                  # Serialised sklearn artefacts (auto-generated)
├── outputs/                      # Saved plots (auto-generated)
├── requirements.txt
└── README.md
```

---

## Data Warehouse (Star Schema)

```
              ┌──────────────┐
              │   Dim_Time   │
              └──────┬───────┘
                     │
┌─────────────┐ ┌────┴──────────────────────┐ ┌───────────────┐
│ Dim_Workout │─│  Fact_Daily_Biometrics     │─│ Dim_Nutrition │
└─────────────┘ │  total_active_minutes     │ └───────────────┘
                │  resting_heart_rate        │
                │  sleep_duration_hours      │
                │  active_calories / steps   │
                │  hrv_score / recovery_score│
                └───────────────────────────┘
```

| Table | Rows | Description |
|-------|------|-------------|
| `Fact_Daily_Biometrics` | 365 | Continuous biometric measurements |
| `Dim_Workout` | 365 | Workout type, intensity, duration |
| `Dim_Nutrition` | 365 | Macros, flags (high-protein, poultry, vegetarian) |
| `Dim_Time` | 365 | Day-of-week, month, season, is_weekend |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data + load into SQLite (auto-run inside notebook too)
python data/generate_data.py
python database/load_data.py

# 3. Open & run the notebook
jupyter notebook notebooks/bimodal_biometric_warehouse.ipynb

# 4. (Optional) Start the Flask REST API (after running the notebook once)
python app/flask_api.py
```

---

## Analysis Pipeline

| Part | Phase | Techniques |
|------|-------|------------|
| 1 | Data Engineering | SQLite star schema, SQLAlchemy, pd.merge |
| 2 | Preprocessing & EDA | Mean imputation, PCA, histograms, heatmaps |
| 3 | Data Mining | K-Means clustering, Apriori rules, Linear Regression, Random Forest + SVM |
| 4 | Evaluation & Deployment | Confusion matrix, F1, ROC-AUC, RMSE, R², Flask API |

---

## Deployment — Flask API

Start the server (requires running the notebook first to generate model artefacts):

```bash
python app/flask_api.py
```

**POST** `http://localhost:5000/predict`

```json
{
  "resting_heart_rate": 62,
  "hrv_score": 58,
  "sleep_duration_hours": 7.2,
  "protein_g": 180,
  "intensity": 7,
  "duration_minutes": 50,
  "total_active_minutes": 45,
  "active_calories": 350,
  "steps": 9500,
  "carbs_g": 210,
  "fat_g": 65,
  "is_high_protein": 1,
  "is_poultry": 1,
  "is_vegetarian": 0,
  "is_weekend": 0
}
```

**Response:**

```json
{
  "recovery_label": 1,
  "recovery_probability": 0.83,
  "message": "Ready to Train",
  "predicted_active_calories": 612.4
}
```