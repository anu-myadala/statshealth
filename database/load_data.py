"""
load_data.py
============
Reads the raw CSVs produced by generate_data.py, performs the integration
merge on ``full_date``, derives dimension flags, assigns surrogate keys, and
bulk-loads the star schema into SQLite.

Surrogate-key assignment is the warehouse's job (not the raw source layer).

Usage:
    python database/load_data.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(BASE_DIR, "database", "biometric_warehouse.db")
SQL_PATH = os.path.join(BASE_DIR, "database", "schema.sql")

SEED = 42
rng  = np.random.default_rng(SEED)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _season(m: int) -> str:
    if m in (12, 1, 2):  return "Winter"
    if m in (3, 4, 5):   return "Spring"
    if m in (6, 7, 8):   return "Summer"
    return "Fall"


def _quarter(m: int) -> int:
    return (m - 1) // 3 + 1


def load():
    # ── (re)create schema ─────────────────────────────────────────────────────
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    with open(SQL_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    engine = create_engine(f"sqlite:///{DB_PATH}")

    # ── read raw sources ──────────────────────────────────────────────────────
    wearable_raw  = pd.read_csv(os.path.join(DATA_DIR, "wearable_raw.csv"))
    nutrition_raw = pd.read_csv(os.path.join(DATA_DIR, "nutrition_raw.csv"))
    workout_sched = pd.read_csv(os.path.join(DATA_DIR, "workout_schedule.csv"))
    print(f"  Raw wearable rows : {len(wearable_raw):,}")
    print(f"  Raw nutrition rows: {len(nutrition_raw):,}")

    # ── aggregate wearable to daily (pd.merge key = full_date) ───────────────
    daily_wearable = (
        wearable_raw
        .groupby("full_date", sort=True)
        .agg(
            steps            =("steps_per_minute", "sum"),
            resting_heart_rate=("heart_rate",      "min"),   # min HR = resting proxy
            mean_heart_rate  =("heart_rate",       "mean"),
            max_heart_rate   =("heart_rate",       "max"),
            active_minutes   =("heart_rate",       lambda x: (x > x.min() + 20).sum()),
        )
        .reset_index()
    )

    # ── aggregate nutrition to daily totals + derive flags ───────────────────
    # Determine dominant meal category per day
    _POULTRY_MEALS    = {"Batch-cooked Chicken Breast", "Ground Turkey Bowl"}
    _VEGETARIAN_MEALS = {"Lentil Soup", "Avocado Toast (Whole Grain)",
                         "Brown Rice + Veggies", "Greek Yogurt + Berries"}

    daily_nutrition = (
        nutrition_raw
        .groupby("full_date", sort=True)
        .agg(
            total_calories=("calories",  "sum"),
            protein_g     =("protein_g", "sum"),
            carbs_g       =("carbs_g",   "sum"),
            fat_g         =("fat_g",     "sum"),
        )
        .reset_index()
    )
    daily_nutrition["is_high_protein"] = (daily_nutrition["protein_g"] > 150).astype(int)

    # Poultry / vegetarian flags: majority of meals in that category?
    daily_nutrition["is_poultry"] = (
        nutrition_raw.groupby("full_date")["meal_name"]
        .apply(lambda s: int(s.isin(_POULTRY_MEALS).mean() >= 0.4))
        .reset_index(drop=True)
    )
    daily_nutrition["is_vegetarian"] = (
        nutrition_raw.groupby("full_date")["meal_name"]
        .apply(lambda s: int(s.isin(_VEGETARIAN_MEALS).mean() >= 0.5))
        .reset_index(drop=True)
    )

    # Derive dominant meal_category label
    def _dominant_category(row):
        if row["is_poultry"]:     return "Batch-cooked Poultry"
        if row["is_vegetarian"]:  return "Vegetarian"
        if row["protein_g"] > 150: return "High Protein"
        if row["carbs_g"] > 250:  return "High-Carb Refuel"
        return "Mixed"
    daily_nutrition["meal_category"] = daily_nutrition.apply(_dominant_category, axis=1)

    # ── merge wearable + nutrition on full_date ───────────────────────────────
    daily = pd.merge(daily_wearable, daily_nutrition, on="full_date", how="inner")
    daily = pd.merge(daily, workout_sched, on="full_date", how="inner")
    daily = daily.sort_values("full_date").reset_index(drop=True)
    print(f"  Merged daily rows : {len(daily)}")

    # ── Dim_Date (surrogate key = row index + 1) ──────────────────────────────
    daily["_dt"] = pd.to_datetime(daily["full_date"])
    dim_date = pd.DataFrame({
        "date_sk":    range(1, len(daily) + 1),
        "full_date":  daily["full_date"].values,
        "day_of_week": daily["_dt"].dt.strftime("%A").values,
        "month":      daily["_dt"].dt.month.values,
        "quarter":    daily["_dt"].dt.month.apply(_quarter).values,
        "season":     daily["_dt"].dt.month.apply(_season).values,
        "is_weekend": (daily["_dt"].dt.weekday >= 5).astype(int).values,
    })

    # ── Dim_Workout (surrogate key assigned here) ─────────────────────────────
    dim_workout = daily[["full_date","workout_type","exercise_category",
                          "intensity","duration_minutes"]].copy()
    dim_workout.insert(0, "workout_sk", range(1, len(dim_workout) + 1))

    # ── Dim_Nutrition (surrogate key assigned here) ───────────────────────────
    dim_nutrition = daily[["full_date","meal_category","total_calories",
                            "protein_g","carbs_g","fat_g",
                            "is_high_protein","is_poultry","is_vegetarian"]].copy()
    dim_nutrition.insert(0, "nutrition_sk", range(1, len(dim_nutrition) + 1))

    # ── Derived biometric columns ─────────────────────────────────────────────
    n = len(daily)
    rng2 = np.random.default_rng(SEED + 1)

    # Sleep: better on rest/yoga days, worse after HIIT
    sleep_effect = np.where(daily["workout_type"] == "Rest", 0.5,
                   np.where(daily["workout_type"] == "HIIT", -0.5, 0.0))
    sleep = np.clip(7.0 + sleep_effect + rng2.normal(0, 0.5, n), 4.5, 9.5).round(1)

    # Active calories
    active_cal = np.clip(
        daily["duration_minutes"].values * daily["intensity"].values * 3.5
        + daily["protein_g"].values * 0.5
        + rng2.normal(0, 40, n),
        0, 1200,
    ).astype(int)

    # HRV
    hrv = np.clip(
        50 + sleep * 3
        - daily["resting_heart_rate"].values * 0.3
        + rng2.normal(0, 5, n),
        20, 100,
    ).round(1)

    # Recovery score & label
    recovery_raw = (
        hrv * 0.4
        + sleep * 4
        + (100 - daily["resting_heart_rate"].values) * 0.3
        - daily["intensity"].values * 1.5
        + daily["is_high_protein"].values * 3
        + rng2.normal(0, 3, n)
    )
    recovery_score = np.clip(recovery_raw, 20, 100).round(1)
    recovery_label = (recovery_score >= 60).astype(int)

    # ── Fact_Daily_Biometrics (surrogate key assigned here) ───────────────────
    fact = pd.DataFrame({
        "fact_sk":             range(1, n + 1),
        "date_sk":             dim_date["date_sk"].values,
        "workout_sk":          dim_workout["workout_sk"].values,
        "nutrition_sk":        dim_nutrition["nutrition_sk"].values,
        "total_active_minutes": daily["active_minutes"].values,
        "resting_heart_rate":  daily["resting_heart_rate"].values.round(1),
        "sleep_duration_hours": sleep,
        "active_calories":     active_cal,
        "steps":               daily["steps"].values,
        "hrv_score":           hrv,
        # PCA components / lag features populated by the notebook after PCA
        "hr_pc1":  None, "hr_pc2":  None, "hr_pc3":  None,
        "hr_pc4":  None, "hr_pc5":  None,
        "lag1_sleep":           None,
        "lag1_active_calories": None,
        "lag1_hrv":             None,
        "recovery_score":      recovery_score,
        "recovery_label":      recovery_label,
    })

    # ── bulk-load ─────────────────────────────────────────────────────────────
    tables = {
        "Dim_Date":              dim_date,
        "Dim_Workout":           dim_workout,
        "Dim_Nutrition":         dim_nutrition,
        "Fact_Daily_Biometrics": fact,
    }
    for tbl, df in tables.items():
        df.to_sql(tbl, engine, if_exists="replace", index=False)
        print(f"  ✓ Loaded {len(df):>4} rows → {tbl}")

    conn.close()
    print(f"\n✓ Database written to {DB_PATH}")
    return daily, dim_date, dim_workout, dim_nutrition, fact


if __name__ == "__main__":
    load()
