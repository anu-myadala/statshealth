"""
preprocessing.py
----------------
Data cleaning, integration, and feature engineering for the biometric warehouse.

Steps
-----
1. Load raw CSVs (wearable + nutrition).
2. Mean-impute missing numeric sensor values.
3. Merge on 'date' using pd.merge.
4. Engineer lag features (next-day RHR, recovery state).
5. Save cleaned dataset to outputs/cleaned_data.csv.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH    = Path(__file__).parent.parent / "database" / "biometric_warehouse.db"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    """Pull a fully-joined analytical dataset from the star schema."""
    query = """
    SELECT
        f.fact_id,
        f.date,
        t.day_of_week,
        t.month,
        t.season,
        t.is_weekend,
        w.workout_type,
        w.workout_intensity,
        w.workout_duration_min,
        w.intensity_score,
        w.is_strength,
        w.is_cardio,
        w.is_rest_day,
        n.meal_category,
        n.total_calories,
        n.protein_g,
        n.carbs_g,
        n.fat_g,
        n.fiber_g,
        n.water_ml,
        n.is_high_protein,
        n.is_poultry,
        n.is_vegetarian,
        n.protein_pct,
        f.steps,
        f.active_minutes,
        f.resting_hr,
        f.sleep_hours,
        f.calories_burned,
        f.recovery_score,
        f.recovery_label
    FROM Fact_Daily_Biometrics f
    JOIN Dim_Time       t ON f.date_id      = t.date_id
    JOIN Dim_Workout    w ON f.workout_id   = w.workout_id
    JOIN Dim_Nutrition  n ON f.nutrition_id = n.nutrition_id
    ORDER BY f.date
    """
    return pd.read_sql_query(query, conn)


def mean_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with column means (sensor gap imputation)."""
    num_cols = df.select_dtypes(include="number").columns
    before   = df[num_cols].isna().sum().sum()
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    after    = df[num_cols].isna().sum().sum()
    print(f"[preprocessing] Imputed {before - after} missing numeric values (mean imputation).")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-day resting HR and previous-day intensity as predictive features."""
    df = df.sort_values("date").reset_index(drop=True)
    df["next_day_rhr"]       = df["resting_hr"].shift(-1)
    df["prev_intensity"]     = df["intensity_score"].shift(1).fillna(0)
    df["rolling_7d_steps"]   = df["steps"].rolling(7, min_periods=1).mean().round(0)
    df["rolling_7d_sleep"]   = df["sleep_hours"].rolling(7, min_periods=1).mean().round(2)
    df["rolling_7d_protein"] = df["protein_g"].rolling(7, min_periods=1).mean().round(1)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode workout intensity for model compatibility."""
    intensity_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    df["workout_intensity_enc"] = df["workout_intensity"].map(intensity_map).fillna(0).astype(int)
    df["recovery_label_enc"]    = (df["recovery_label"] == "Ready to Train").astype(int)
    return df


def run_preprocessing() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df   = load_from_db(conn)
    conn.close()

    print(f"[preprocessing] Loaded {len(df)} rows from database.")

    df = mean_impute(df)
    df = add_lag_features(df)
    df = encode_categoricals(df)

    out_path = OUTPUT_DIR / "cleaned_data.csv"
    df.to_csv(out_path, index=False)
    print(f"[preprocessing] Cleaned dataset saved → {out_path}")
    return df


if __name__ == "__main__":
    run_preprocessing()
