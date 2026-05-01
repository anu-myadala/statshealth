"""
generate_data.py
----------------
Generates synthetic wearable (Fitbit/Apple Watch) and nutritional (MyFitnessPal)
data for 365 days to support the Bimodal Biometric Warehouse project.

Outputs
-------
data/wearable_data.csv   – kinematics, heart rate, sleep, workout logs
data/nutrition_data.csv  – daily macro-nutrient logs
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

N_DAYS = 365
START_DATE = "2024-01-01"

WORKOUT_TYPES = ["Dumbbell Strength", "Incline Walk", "Rest", "HIIT", "Yoga"]
WORKOUT_WEIGHTS = [0.25, 0.20, 0.30, 0.15, 0.10]

MEAL_CATEGORIES = [
    "Batch-cooked Poultry",
    "Vegetarian",
    "Mixed Macros",
    "High-Carb Refeed",
    "Low-Calorie Deficit",
]
MEAL_WEIGHTS = [0.30, 0.15, 0.30, 0.15, 0.10]


def _workout_stats(workout_type: str) -> dict:
    """Return plausible workout duration and intensity for a given type."""
    stats = {
        "Dumbbell Strength": {"duration_min": rng.integers(45, 75), "intensity": rng.choice(["Medium", "High"], p=[0.4, 0.6])},
        "Incline Walk":      {"duration_min": rng.integers(30, 60), "intensity": rng.choice(["Low", "Medium"], p=[0.5, 0.5])},
        "Rest":              {"duration_min": 0,                    "intensity": "None"},
        "HIIT":              {"duration_min": rng.integers(20, 40), "intensity": "High"},
        "Yoga":              {"duration_min": rng.integers(30, 60), "intensity": "Low"},
    }
    return stats[workout_type]


def generate_wearable(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    prev_rhr = 62.0
    for date in dates:
        workout_type = rng.choice(WORKOUT_TYPES, p=WORKOUT_WEIGHTS)
        ws = _workout_stats(workout_type)

        intensity_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
        intensity_score = intensity_map[ws["intensity"]]

        # Steps: rest days ~3 k, active days up to ~15 k
        steps = int(
            rng.integers(2_000, 4_000)
            + intensity_score * rng.integers(2_000, 4_000)
        )

        # Active minutes correlate with duration
        active_minutes = max(0, int(ws["duration_min"] + rng.normal(0, 5)))

        # RHR: mean-reverts around 62, intensity raises it slightly next day
        delta_rhr = rng.normal(0, 1.5) + 0.8 * intensity_score
        rhr = round(np.clip(prev_rhr + delta_rhr * 0.3, 48, 85), 1)
        prev_rhr = rhr

        # Sleep: harder sessions slightly reduce sleep quality
        sleep_hours = round(
            np.clip(rng.normal(7.2, 0.8) - 0.15 * intensity_score, 4.0, 10.0), 2
        )

        # Calories burned: baseline + workout contribution
        calories_burned = int(
            rng.normal(1_800, 150)
            + intensity_score * rng.integers(100, 350)
            + steps * 0.04
        )

        rows.append(
            {
                "date": date.date(),
                "steps": steps,
                "active_minutes": active_minutes,
                "resting_hr": rhr,
                "sleep_hours": sleep_hours,
                "calories_burned": calories_burned,
                "workout_type": workout_type,
                "workout_intensity": ws["intensity"],
                "workout_duration_min": ws["duration_min"],
            }
        )
    return pd.DataFrame(rows)


def generate_nutrition(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for date in dates:
        meal_category = rng.choice(MEAL_CATEGORIES, p=MEAL_WEIGHTS)

        # Macro profiles per category
        profiles = {
            "Batch-cooked Poultry":  {"cal": (2_100, 200), "prot": (180, 20),  "carb": (160, 25), "fat": (55, 10)},
            "Vegetarian":            {"cal": (1_900, 180), "prot": (90, 15),   "carb": (240, 30), "fat": (60, 10)},
            "Mixed Macros":          {"cal": (2_200, 250), "prot": (130, 25),  "carb": (210, 35), "fat": (70, 15)},
            "High-Carb Refeed":      {"cal": (2_600, 300), "prot": (110, 20),  "carb": (380, 40), "fat": (55, 12)},
            "Low-Calorie Deficit":   {"cal": (1_500, 150), "prot": (140, 20),  "carb": (120, 20), "fat": (45, 8)},
        }
        p = profiles[meal_category]
        total_calories  = max(800,  int(rng.normal(*p["cal"])))
        protein_g       = max(40,   int(rng.normal(*p["prot"])))
        carbs_g         = max(50,   int(rng.normal(*p["carb"])))
        fat_g           = max(20,   int(rng.normal(*p["fat"])))
        fiber_g         = max(5,    int(rng.normal(22, 6)))
        water_ml        = max(500,  int(rng.normal(2_400, 400)))

        rows.append(
            {
                "date": date.date(),
                "total_calories": total_calories,
                "protein_g": protein_g,
                "carbs_g": carbs_g,
                "fat_g": fat_g,
                "fiber_g": fiber_g,
                "water_ml": water_ml,
                "meal_category": meal_category,
            }
        )
    return pd.DataFrame(rows)


def introduce_missing(df: pd.DataFrame, frac: float = 0.03) -> pd.DataFrame:
    """Randomly null-out a small fraction of numeric cells to simulate sensor gaps."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan
    return df


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

    wearable = generate_wearable(dates)
    wearable = introduce_missing(wearable)

    nutrition = generate_nutrition(dates)
    nutrition = introduce_missing(nutrition)

    wearable.to_csv(out_dir / "wearable_data.csv", index=False)
    nutrition.to_csv(out_dir / "nutrition_data.csv", index=False)

    print(f"[generate_data] Saved {len(wearable)} rows → wearable_data.csv")
    print(f"[generate_data] Saved {len(nutrition)} rows → nutrition_data.csv")


if __name__ == "__main__":
    main()
