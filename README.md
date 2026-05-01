# The Bimodal Biometric Warehouse
## Mining Wearable and Nutritional Data for Metabolic Optimization

A complete end-to-end data science project that builds a **relational star-schema data warehouse** from synthetic wearable (Fitbit/Apple Watch) and nutritional (MyFitnessPal) data, then applies four major data-mining techniques to model metabolic health and recovery for "bimodal" activity profiles.

---

## Project Overview

Most fitness models assume a consistent daily activity level.  This project targets **bimodal** activity profiles — days characterised by short bursts of intense exercise (dumbbell training, high-incline walking) followed by highly sedentary periods — and asks:

* **Can we predict exact caloric expenditure** from nutrition and workout features?
* **Can we classify tomorrow's recovery state** ("Ready to Train" / "Needs Rest")?
* **What hidden lifestyle clusters** emerge from daily biometric logs?
* **What nutrition–workout combinations** are statistically linked to deeper sleep?

---

## Repository Structure

```
statshealth/
├── main.py                         # End-to-end orchestration (run this)
├── requirements.txt
│
├── data/
│   └── generate_data.py            # Synthetic wearable + nutrition CSV generation
│
├── database/
│   └── setup_db.py                 # SQLite star-schema builder
│
├── analysis/
│   ├── preprocessing.py            # Cleaning, imputation, feature engineering
│   ├── pca_analysis.py             # PCA on HR feature space
│   └── eda.py                      # Histograms, heatmaps, scatter plots
│
├── models/
│   ├── clustering.py               # K-Means clustering
│   ├── association_rules.py        # Apriori association rule mining
│   ├── regression.py               # Multi-variable Linear Regression
│   └── classification.py           # Random Forest + SVM classification
│
├── evaluation/
│   └── evaluate.py                 # Consolidated metrics report
│
└── app/
    └── flask_app.py                # Flask prediction API
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (generates data → trains models → evaluates)
python main.py

# 3. (Optional) Launch the prediction API
python app/flask_app.py
```

All plots and reports are written to `outputs/`.

---

## Part 1 – Data Engineering & Warehousing

### Data Sources (Simulated)

| Source | Description |
|---|---|
| `wearable_data.csv` | 365 days of Fitbit/Apple Watch exports: steps, active minutes, resting HR, sleep hours, calories burned, workout type & intensity |
| `nutrition_data.csv` | 365 days of MyFitnessPal logs: total calories, protein/carbs/fat/fiber (g), water intake, meal category |

### Star Schema (SQLite)

```
Dim_Time ──────────────┐
                        │
Dim_Workout ────────── Fact_Daily_Biometrics
                        │
Dim_Nutrition ──────────┘
```

| Table | Key Columns |
|---|---|
| **Fact_Daily_Biometrics** | fact_id, date_id, workout_id, nutrition_id, steps, active_minutes, resting_hr, sleep_hours, calories_burned, recovery_label |
| **Dim_Time** | date_id, date, day_of_week, month, season, is_weekend |
| **Dim_Workout** | workout_id, workout_type, intensity, duration_min, is_strength, is_cardio, is_rest_day |
| **Dim_Nutrition** | nutrition_id, meal_category, protein_g, carbs_g, fat_g, is_high_protein, is_vegetarian, is_poultry |

---

## Part 2 – Preprocessing, EDA & Visualization

* **Missing-value imputation**: column-mean imputation for ~3 % sensor gaps.
* **Feature engineering**: 7-day rolling averages (steps, sleep, protein), lag features (next-day RHR, prior-day intensity).
* **PCA**: 20 simulated HR-zone features compressed to 5 principal components (PC1 + PC2 capture > 73 % variance).
* **EDA plots** (saved to `outputs/plots/`):

| Plot | Description |
|---|---|
| `eda_distributions.png` | Histograms of all key biometric & nutritional metrics |
| `eda_correlation_hmap.png` | Full correlation heat-map |
| `eda_protein_vs_rhr.png` | High-protein intake days correlated with next-day resting HR |
| `eda_workout_sleep.png` | Sleep duration by workout type (box plots) |
| `eda_weekly_pattern.png` | Mean calorie burn by day of week |
| `pca_variance.png` | Explained variance per principal component |

---

## Part 3 – Data Mining

### K-Means Clustering

Daily logs clustered into **3 lifestyle profiles**:

| Cluster | Description |
|---|---|
| High Strain / Low Recovery | High steps, high intensity, elevated resting HR |
| Optimal Balance | Moderate activity, healthy HR + sleep |
| Sedentary | Low steps, near-zero active minutes, rest days |

Outputs: `outputs/cluster_labels.csv`, `outputs/plots/kmeans_clusters.png`, `outputs/plots/kmeans_elbow.png`

### Apriori Association Rule Mining

Transactions encoded from meal category, workout type, intensity, sleep quality, and recovery label.

Example rules discovered (lift > 5×):

```
{Workout=Dumbbell Strength, Meal=Batch-cooked Poultry} → {HighProtein=Yes, Intensity=High}   conf=1.00  lift=5.37
{Workout=Yoga}                                          → {Recovery=Ready to Train, Intensity=Low}  conf=0.59  lift=5.42
```

Outputs: `outputs/association_rules.csv`, `outputs/plots/arules_scatter.png`

### Multi-variable Linear Regression

**Target**: `calories_burned`  
**Features**: protein_g, intensity_score, active_minutes, steps, sleep_hours, is_high_protein, carbs_g, fat_g

| Metric | Value |
|---|---|
| Test RMSE | ~222 kcal |
| Test R² | ~0.76 |
| 5-fold CV R² | ~0.76 |

Outputs: `outputs/regression_results.txt`, `outputs/plots/regression_*.png`

### Random Forest + SVM Classification

**Target**: `recovery_label` — "Ready to Train" / "Needs Rest"

| Model | F1 (weighted) | ROC-AUC |
|---|---|---|
| Random Forest | ~0.89 | ~0.98 |
| SVM (RBF) | ~0.82 | ~0.96 |

Outputs: `outputs/classification_report.txt`, `outputs/plots/confusion_matrix_*.png`, `outputs/plots/roc_curve.png`, `outputs/plots/rf_feature_importance.png`

---

## Part 4 – Evaluation & Deployment

### Evaluation Metrics

```
Regression  : RMSE, R²
Classification : Confusion Matrix, F1 Score, ROC-AUC
```

Full report: `outputs/evaluation_summary.txt`

### Flask Prediction API

```bash
python app/flask_app.py   # starts on http://0.0.0.0:5000
```

**Endpoints**

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/predict/recovery` | Predict next-day recovery state + confidence |
| POST | `/predict/calories` | Predict calorie expenditure |

**Example – Recovery Prediction**

```bash
curl -X POST http://localhost:5000/predict/recovery \
     -H "Content-Type: application/json" \
     -d '{
           "steps": 9000, "active_minutes": 55, "resting_hr": 62,
           "sleep_hours": 7.5, "calories_burned": 2400,
           "intensity_score": 2, "protein_g": 160,
           "carbs_g": 180, "fat_g": 60,
           "is_high_protein": 1, "is_strength": 1, "is_cardio": 0,
           "rolling_7d_steps": 8000, "rolling_7d_sleep": 7.2,
           "rolling_7d_protein": 145, "prev_intensity": 1
         }'
```

```json
{
  "recovery_label": "Ready to Train",
  "recovery_label_enc": 1,
  "confidence": 0.885
}
```

---

## Technologies Used

| Library | Purpose |
|---|---|
| pandas / numpy | Data manipulation |
| scikit-learn | PCA, K-Means, Linear Regression, Random Forest, SVM |
| mlxtend | Apriori association rule mining |
| matplotlib / seaborn | Visualisations |
| SQLite / SQLAlchemy | Star-schema data warehouse |
| Flask | Prediction REST API |
| joblib | Model serialisation |
