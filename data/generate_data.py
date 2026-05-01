"""
generate_data.py
Generates synthetic wearable (Fitbit-style) and nutritional (MyFitnessPal-style)
CSV data for 365 days to simulate a "bimodal" activity profile.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

SEED = 42
rng = np.random.default_rng(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────

def _clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


# ── date spine ───────────────────────────────────────────────────────────────

start = date(2024, 1, 1)
dates = [start + timedelta(days=i) for i in range(365)]
n = len(dates)

# ── workout dimension ─────────────────────────────────────────────────────────

workout_types = rng.choice(
    ["Dumbbell Strength", "Incline Walk", "HIIT", "Rest", "Yoga / Stretch"],
    size=n,
    p=[0.25, 0.25, 0.15, 0.25, 0.10],
)

intensity_map = {
    "Dumbbell Strength": (6, 9),
    "Incline Walk":      (5, 8),
    "HIIT":              (8, 10),
    "Rest":              (1, 2),
    "Yoga / Stretch":    (2, 4),
}

intensity = np.array([
    rng.integers(*intensity_map[w]) for w in workout_types
])

duration = np.where(
    workout_types == "Rest",
    0,
    _clamp(rng.normal(45, 15, n).astype(int), 20, 90),
)

category_map = {
    "Dumbbell Strength": "Strength",
    "Incline Walk":      "Cardio",
    "HIIT":              "Cardio",
    "Rest":              "Rest",
    "Yoga / Stretch":    "Recovery",
}
exercise_category = np.array([category_map[w] for w in workout_types])

workout_df = pd.DataFrame({
    "workout_id":        range(1, n + 1),
    "workout_type":      workout_types,
    "exercise_category": exercise_category,
    "intensity":         intensity,
    "duration_minutes":  duration,
})

# ── nutrition dimension ───────────────────────────────────────────────────────

meal_categories = rng.choice(
    ["Batch-cooked Poultry", "Mixed Protein", "Vegetarian", "High-Carb Refuel", "Cheat Day"],
    size=n,
    p=[0.30, 0.25, 0.20, 0.15, 0.10],
)

protein_g = _clamp(
    np.where(
        meal_categories == "Batch-cooked Poultry",
        rng.normal(185, 20, n),
        np.where(
            meal_categories == "Mixed Protein",
            rng.normal(150, 20, n),
            np.where(
                meal_categories == "Vegetarian",
                rng.normal(100, 20, n),
                np.where(
                    meal_categories == "High-Carb Refuel",
                    rng.normal(120, 20, n),
                    rng.normal(90, 20, n),   # Cheat Day
                ),
            ),
        ),
    ),
    60, 250,
).astype(int)

carbs_g = _clamp(rng.normal(200, 50, n), 80, 400).astype(int)
fat_g   = _clamp(rng.normal(70, 20, n),  30, 150).astype(int)
total_calories = (protein_g * 4 + carbs_g * 4 + fat_g * 9).astype(int)

is_high_protein  = (protein_g > 150).astype(int)
is_poultry       = (meal_categories == "Batch-cooked Poultry").astype(int)
is_vegetarian    = (meal_categories == "Vegetarian").astype(int)

nutrition_df = pd.DataFrame({
    "nutrition_id":   range(1, n + 1),
    "meal_category":  meal_categories,
    "total_calories": total_calories,
    "protein_g":      protein_g,
    "carbs_g":        carbs_g,
    "fat_g":          fat_g,
    "is_high_protein": is_high_protein,
    "is_poultry":      is_poultry,
    "is_vegetarian":   is_vegetarian,
})

# ── time dimension ────────────────────────────────────────────────────────────

day_names  = [d.strftime("%A") for d in dates]
months     = [d.month for d in dates]
is_weekend = [1 if d.weekday() >= 5 else 0 for d in dates]

def season(m):
    if m in (12, 1, 2):  return "Winter"
    if m in (3, 4, 5):   return "Spring"
    if m in (6, 7, 8):   return "Summer"
    return "Fall"

seasons = [season(m) for m in months]

time_df = pd.DataFrame({
    "time_id":    range(1, n + 1),
    "date":       [str(d) for d in dates],
    "day_of_week": day_names,
    "month":      months,
    "season":     seasons,
    "is_weekend": is_weekend,
})

# ── fact table ────────────────────────────────────────────────────────────────

# Active minutes influenced by workout intensity + duration
base_active = duration * (intensity / 10.0)
total_active_minutes = _clamp(
    (base_active + rng.normal(0, 5, n)).astype(int), 0, 120
)

# Resting HR: goes down with cardio/strength days, up on rest days
rhr_base = 62
rhr_noise = rng.normal(0, 3, n)
rhr_workout_effect = np.where(
    exercise_category == "Cardio", -3,
    np.where(exercise_category == "Strength", -2, 0)
)
# High-protein reduces RHR slightly next day (shift by 1)
protein_rhr_effect = np.zeros(n)
protein_rhr_effect[1:] = np.where(is_high_protein[:-1], -1.5, 0)

resting_hr = _clamp(
    (rhr_base + rhr_noise + rhr_workout_effect + protein_rhr_effect).astype(int),
    45, 90,
)

# Sleep: better on rest/yoga days, worse after HIIT
sleep_base = 7.0
sleep_effect = np.where(
    workout_types == "Rest", 0.5,
    np.where(workout_types == "HIIT", -0.5, 0.0)
)
sleep_duration = _clamp(
    sleep_base + sleep_effect + rng.normal(0, 0.5, n), 4.5, 9.5
).round(1)

# Active calories: driven by intensity × duration, modulated by protein
active_calories = _clamp(
    (duration * intensity * 3.5 + protein_g * 0.5 + rng.normal(0, 40, n)).astype(int),
    0, 1200,
)

# Steps
steps = _clamp(
    np.where(
        exercise_category == "Cardio",
        rng.integers(8000, 18000, n),
        np.where(
            exercise_category == "Strength",
            rng.integers(5000, 10000, n),
            rng.integers(2000, 7000, n),
        ),
    ),
    1000, 25000,
)

# HRV score (higher = better recovery)
hrv = _clamp(
    (50 + sleep_duration * 3 - resting_hr * 0.3 + rng.normal(0, 5, n)).astype(int),
    20, 100,
)

# Recovery score (target for classification)
recovery_raw = (
    hrv * 0.4
    + sleep_duration * 4
    + (100 - resting_hr) * 0.3
    - intensity * 1.5
    + is_high_protein * 3
    + rng.normal(0, 3, n)
)
recovery_score = _clamp(recovery_raw.astype(int), 20, 100)

# Binary label: 1 = "Ready to Train", 0 = "Needs Rest"
recovery_label = (recovery_score >= 60).astype(int)

fact_df = pd.DataFrame({
    "fact_id":             range(1, n + 1),
    "date_id":             range(1, n + 1),   # FK → Dim_Time
    "workout_id":          range(1, n + 1),   # FK → Dim_Workout
    "nutrition_id":        range(1, n + 1),   # FK → Dim_Nutrition
    "time_id":             range(1, n + 1),   # FK → Dim_Time
    "total_active_minutes": total_active_minutes,
    "resting_heart_rate":  resting_hr,
    "sleep_duration_hours": sleep_duration,
    "active_calories":     active_calories,
    "steps":               steps,
    "hrv_score":           hrv,
    "recovery_score":      recovery_score,
    "recovery_label":      recovery_label,   # 1=Ready, 0=Needs Rest
})

# ── persist ───────────────────────────────────────────────────────────────────

out_dir = os.path.dirname(__file__)
workout_df.to_csv(os.path.join(out_dir, "dim_workout.csv"), index=False)
nutrition_df.to_csv(os.path.join(out_dir, "dim_nutrition.csv"), index=False)
time_df.to_csv(os.path.join(out_dir, "dim_time.csv"), index=False)
fact_df.to_csv(os.path.join(out_dir, "fact_daily_biometrics.csv"), index=False)

print("✓ Generated 365-day synthetic dataset")
print(f"  Workout types   : {dict(pd.Series(workout_types).value_counts())}")
print(f"  Meal categories : {dict(pd.Series(meal_categories).value_counts())}")
print(f"  Ready-to-train  : {recovery_label.sum()} / {n} days")
