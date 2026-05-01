"""
setup_db.py
-----------
Builds an SQLite star-schema database from the wearable and nutrition CSV files.

Star Schema
-----------
Dim_Time          – date, day_of_week, month, season, is_weekend
Dim_Workout       – workout type, intensity, duration, flags
Dim_Nutrition     – daily macros, categorical flags
Fact_Daily_Biometrics – joins dim keys + continuous measures

Usage
-----
    python database/setup_db.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

DB_PATH = Path(__file__).parent / "biometric_warehouse.db"
DATA_DIR = Path(__file__).parent.parent / "data"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: season from month
# ──────────────────────────────────────────────────────────────────────────────
def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


# ──────────────────────────────────────────────────────────────────────────────
# Build dimension tables
# ──────────────────────────────────────────────────────────────────────────────
def build_dim_time(dates: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    df["date_id"]    = range(1, len(df) + 1)
    df["day_of_week"] = df["date"].dt.day_name()
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.month_name()
    df["season"]      = df["month"].apply(_season)
    df["is_weekend"]  = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["date"]        = df["date"].dt.date.astype(str)
    return df[["date_id", "date", "day_of_week", "month", "month_name", "season", "is_weekend"]]


def build_dim_workout(wearable: pd.DataFrame) -> pd.DataFrame:
    intensity_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    df = wearable[["workout_type", "workout_intensity", "workout_duration_min"]].copy()
    df["workout_id"]       = range(1, len(df) + 1)
    df["intensity_score"]  = df["workout_intensity"].map(intensity_map).fillna(0).astype(int)
    df["is_strength"]      = (df["workout_type"] == "Dumbbell Strength").astype(int)
    df["is_cardio"]        = df["workout_type"].isin(["Incline Walk", "HIIT"]).astype(int)
    df["is_rest_day"]      = (df["workout_type"] == "Rest").astype(int)
    return df[["workout_id", "workout_type", "workout_intensity", "workout_duration_min",
               "intensity_score", "is_strength", "is_cardio", "is_rest_day"]]


KCAL_PER_GRAM_PROTEIN = 4  # Atwater factor: 4 kcal per gram of protein


def build_dim_nutrition(nutrition: pd.DataFrame) -> pd.DataFrame:
    df = nutrition[["meal_category", "total_calories", "protein_g",
                    "carbs_g", "fat_g", "fiber_g", "water_ml"]].copy()
    df["nutrition_id"]    = range(1, len(df) + 1)
    df["is_high_protein"] = (df["protein_g"] >= 150).astype(int)
    df["is_poultry"]      = (df["meal_category"] == "Batch-cooked Poultry").astype(int)
    df["is_vegetarian"]   = (df["meal_category"] == "Vegetarian").astype(int)
    df["protein_pct"]     = (df["protein_g"] * KCAL_PER_GRAM_PROTEIN / df["total_calories"].replace(0, np.nan) * 100).round(1)
    return df[["nutrition_id", "meal_category", "total_calories", "protein_g", "carbs_g",
               "fat_g", "fiber_g", "water_ml", "is_high_protein", "is_poultry",
               "is_vegetarian", "protein_pct"]]


def build_fact_table(
    wearable: pd.DataFrame,
    nutrition: pd.DataFrame,
    dim_time: pd.DataFrame,
    dim_workout: pd.DataFrame,
    dim_nutrition: pd.DataFrame,
) -> pd.DataFrame:
    # attach surrogate keys
    w = wearable.copy().reset_index(drop=True)
    n = nutrition.copy().reset_index(drop=True)

    w["date_id"]      = dim_time["date_id"].values
    w["workout_id"]   = dim_workout["workout_id"].values
    n["nutrition_id"] = dim_nutrition["nutrition_id"].values

    fact = pd.merge(w, n[["date", "nutrition_id"]], on="date", how="inner")
    fact["fact_id"] = range(1, len(fact) + 1)

    # Derive next-day recovery label: "Ready to Train" / "Needs Rest"
    # Based on RHR (lower = better), sleep > 7 h, and workout intensity
    fact["recovery_score"] = (
        (65 - fact["resting_hr"].fillna(fact["resting_hr"].mean())) * 0.4
        + (fact["sleep_hours"].fillna(fact["sleep_hours"].mean()) - 6) * 1.5
        - fact["workout_id"].map(dim_workout.set_index("workout_id")["intensity_score"]).fillna(1) * 0.8
    )
    median_rs = fact["recovery_score"].median()
    fact["recovery_label"] = (fact["recovery_score"] >= median_rs).map(
        {True: "Ready to Train", False: "Needs Rest"}
    )

    cols = [
        "fact_id", "date", "date_id", "workout_id", "nutrition_id",
        "steps", "active_minutes", "resting_hr", "sleep_hours",
        "calories_burned", "recovery_score", "recovery_label",
    ]
    return fact[cols]


# ──────────────────────────────────────────────────────────────────────────────
# Write to SQLite
# ──────────────────────────────────────────────────────────────────────────────
def load_to_db(
    conn: sqlite3.Connection,
    dim_time: pd.DataFrame,
    dim_workout: pd.DataFrame,
    dim_nutrition: pd.DataFrame,
    fact: pd.DataFrame,
) -> None:
    dim_time.to_sql("Dim_Time",       conn, if_exists="replace", index=False)
    dim_workout.to_sql("Dim_Workout",  conn, if_exists="replace", index=False)
    dim_nutrition.to_sql("Dim_Nutrition", conn, if_exists="replace", index=False)
    fact.to_sql("Fact_Daily_Biometrics",  conn, if_exists="replace", index=False)

    # Add primary / foreign key metadata (informational – SQLite doesn't enforce FK by default)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_fact_date_id      ON Fact_Daily_Biometrics(date_id);
        CREATE INDEX IF NOT EXISTS idx_fact_workout_id   ON Fact_Daily_Biometrics(workout_id);
        CREATE INDEX IF NOT EXISTS idx_fact_nutrition_id ON Fact_Daily_Biometrics(nutrition_id);
        """
    )
    conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    wearable  = pd.read_csv(DATA_DIR / "wearable_data.csv")
    nutrition = pd.read_csv(DATA_DIR / "nutrition_data.csv")

    dim_time      = build_dim_time(wearable["date"])
    dim_workout   = build_dim_workout(wearable)
    dim_nutrition = build_dim_nutrition(nutrition)
    fact          = build_fact_table(wearable, nutrition, dim_time, dim_workout, dim_nutrition)

    conn = sqlite3.connect(DB_PATH)
    load_to_db(conn, dim_time, dim_workout, dim_nutrition, fact)
    conn.close()

    print(f"[setup_db] Database written to {DB_PATH}")
    print(f"  Dim_Time        : {len(dim_time)} rows")
    print(f"  Dim_Workout     : {len(dim_workout)} rows")
    print(f"  Dim_Nutrition   : {len(dim_nutrition)} rows")
    print(f"  Fact_Daily_Biometrics : {len(fact)} rows")


if __name__ == "__main__":
    main()
