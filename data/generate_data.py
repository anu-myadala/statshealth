"""
generate_data.py
================
Generates two distinct raw data sources that simulate real-world exports:

  wearable_raw.csv   — minute-level heart-rate + steps (Fitbit-style)
                       365 days × 1440 minutes = 525,600 rows
  nutrition_raw.csv  — per-meal nutrition log (MyFitnessPal-style)
                       3–5 meal entries per day, one row each

These two sources are intentionally kept separate and later merged in the
notebook on the ``full_date`` key, mimicking a real integration pipeline.

Surrogate keys (workout_sk, nutrition_sk, date_sk) are assigned during the
dimension-building step in load_data.py — NOT here — so this script stays
a "raw source" layer with no warehouse concerns.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

SEED = 42
rng  = np.random.default_rng(SEED)

N_DAYS = 365
START  = date(2024, 1, 1)
DATES  = [START + timedelta(days=i) for i in range(N_DAYS)]
DATE_STRS = [str(d) for d in DATES]


# ─── workout schedule (drives HR waveform shape) ─────────────────────────────

_INTENSITY_MAP = {
    "Dumbbell Strength": (6, 9),
    "Incline Walk":      (5, 8),
    "HIIT":              (8, 10),
    "Rest":              (1, 2),
    "Yoga / Stretch":    (2, 4),
}
_CATEGORY_MAP = {
    "Dumbbell Strength": "Strength",
    "Incline Walk":      "Cardio",
    "HIIT":              "Cardio",
    "Rest":              "Rest",
    "Yoga / Stretch":    "Recovery",
}

WORKOUT_TYPES = rng.choice(
    list(_INTENSITY_MAP.keys()), size=N_DAYS,
    p=[0.25, 0.25, 0.15, 0.25, 0.10],
)
INTENSITY = np.array([rng.integers(*_INTENSITY_MAP[w]) for w in WORKOUT_TYPES])
DURATION  = np.where(
    WORKOUT_TYPES == "Rest", 0,
    np.clip(rng.normal(45, 15, N_DAYS).astype(int), 20, 90),
)
CATEGORY = np.array([_CATEGORY_MAP[w] for w in WORKOUT_TYPES])

# Per-day resting HR base (individual variation)
RESTING_HR_BASE = np.clip(rng.normal(64, 4, N_DAYS).astype(int), 50, 82)


# ─── SOURCE 1: minute-level wearable export ───────────────────────────────────

def _build_day_hr(rhr: int, wtype: str, intens: int, dur: int) -> np.ndarray:
    """Return a 1440-element float array of minute-by-minute HR for one day.

    The waveform is deliberately *bimodal*:
      • Peak 1 — morning structured workout (cardio or strength spike)
      • Peak 2 — short afternoon NEAT walk (~30 min around 2 pm)
    Rest / Yoga days produce a flat near-resting baseline, giving PCA a clear
    contrast between bimodal and sedentary signatures.
    """
    hr = rng.normal(rhr, 2.0, 1440)           # baseline + noise

    if wtype == "Rest":
        return np.clip(hr, 40, 120).round(1)

    # ── Peak 1: morning workout (5–7 am = minutes 300–420) ────────────────────
    w_start  = int(rng.uniform(300, 420))
    hr_peak  = rhr + intens * 8 + rng.normal(0, 4)
    for m in range(w_start, min(w_start + dur, 1440)):
        t        = (m - w_start) / dur
        envelope = np.sin(np.pi * t)          # smooth rise-and-fall
        hr[m]    = rhr + (hr_peak - rhr) * envelope + rng.normal(0, 3)

    # ── Peak 2: afternoon NEAT walk (1–3 pm = minutes 780–900) ───────────────
    if wtype != "Yoga / Stretch":             # yoga already low-intensity
        a_start = int(rng.uniform(780, 900))
        a_dur   = int(rng.uniform(15, 40))
        a_peak  = rhr + 15 + rng.normal(0, 5)
        for m in range(a_start, min(a_start + a_dur, 1440)):
            t        = (m - a_start) / a_dur
            envelope = np.sin(np.pi * t)
            hr[m]    = max(hr[m], rhr + (a_peak - rhr) * envelope)

    return np.clip(hr, 40, 200).round(1)


print("Building minute-level wearable data (525,600 rows) …")
day_frames = []
for i, (d, wt, ins, dur, rhr) in enumerate(
        zip(DATE_STRS, WORKOUT_TYPES, INTENSITY, DURATION, RESTING_HR_BASE)):
    hr_arr   = _build_day_hr(rhr, wt, ins, dur)
    # Steps: proportional to HR above resting, zero when near resting
    steps_per_min = np.where(
        hr_arr > rhr + 20, rng.integers(8, 22, 1440),
        np.where(hr_arr > rhr + 5, rng.integers(1, 8, 1440), 0),
    )
    day_frames.append(pd.DataFrame({
        "full_date":        d,
        "minute":           np.arange(1440),
        "heart_rate":       hr_arr,
        "steps_per_minute": steps_per_min,
    }))

wearable_raw = pd.concat(day_frames, ignore_index=True)
print(f"  wearable_raw shape: {wearable_raw.shape}")


# ─── SOURCE 2: per-meal nutrition log ────────────────────────────────────────

_MEAL_TEMPLATES = {
    "Batch-cooked Chicken Breast": dict(
        cal=(380, 480), prot=(42, 55), carb=(8, 20),  fat=(6, 14)),
    "Ground Turkey Bowl":          dict(
        cal=(420, 520), prot=(38, 50), carb=(30, 50),  fat=(10, 20)),
    "Protein Shake":               dict(
        cal=(140, 200), prot=(25, 35), carb=(5, 15),   fat=(2, 6)),
    "Greek Yogurt + Berries":      dict(
        cal=(180, 250), prot=(15, 22), carb=(20, 35),  fat=(3, 8)),
    "Egg White Omelette":          dict(
        cal=(200, 300), prot=(20, 30), carb=(5, 15),   fat=(4, 10)),
    "Brown Rice + Veggies":        dict(
        cal=(300, 400), prot=(8, 15),  carb=(55, 75),  fat=(4, 10)),
    "Avocado Toast (Whole Grain)": dict(
        cal=(350, 450), prot=(10, 18), carb=(40, 55),  fat=(14, 22)),
    "Lentil Soup":                 dict(
        cal=(280, 380), prot=(14, 22), carb=(40, 55),  fat=(5, 12)),
    "Cheeseburger + Fries":        dict(
        cal=(700, 950), prot=(28, 40), carb=(70, 95),  fat=(30, 50)),
    "Mixed Nuts (Snack)":          dict(
        cal=(160, 220), prot=(5, 9),   carb=(6, 12),   fat=(13, 19)),
    "Incline Walk Fuel Bar":       dict(
        cal=(200, 280), prot=(10, 18), carb=(28, 40),  fat=(5, 12)),
    "Post-Workout Rice Cakes":     dict(
        cal=(120, 180), prot=(3, 7),   carb=(25, 38),  fat=(1, 4)),
}
MEAL_NAMES = list(_MEAL_TEMPLATES.keys())

# Assign a "dominant meal pattern" per day (drives which meals appear)
_PATTERN_MEAL_POOL = {
    "High Protein": ["Batch-cooked Chicken Breast", "Ground Turkey Bowl",
                     "Protein Shake", "Egg White Omelette", "Greek Yogurt + Berries"],
    "Carb Refuel":  ["Brown Rice + Veggies", "Avocado Toast (Whole Grain)",
                     "Post-Workout Rice Cakes", "Lentil Soup", "Incline Walk Fuel Bar"],
    "Vegetarian":   ["Lentil Soup", "Avocado Toast (Whole Grain)", "Brown Rice + Veggies",
                     "Greek Yogurt + Berries", "Mixed Nuts (Snack)"],
    "Cheat Day":    ["Cheeseburger + Fries", "Mixed Nuts (Snack)", "Protein Shake"],
    "Mixed":        MEAL_NAMES,
}
PATTERN = rng.choice(
    list(_PATTERN_MEAL_POOL.keys()), size=N_DAYS,
    p=[0.30, 0.20, 0.20, 0.10, 0.20],
)

meal_rows = []
for d_str, pat in zip(DATE_STRS, PATTERN):
    pool   = _PATTERN_MEAL_POOL[pat]
    n_meals = int(rng.integers(3, 6))
    chosen  = rng.choice(pool, size=n_meals, replace=True)
    for meal in chosen:
        t = _MEAL_TEMPLATES[meal]
        meal_rows.append({
            "full_date":  d_str,
            "meal_name":  meal,
            "calories":   int(rng.integers(t["cal"][0], t["cal"][1])),
            "protein_g":  int(rng.integers(t["prot"][0], t["prot"][1])),
            "carbs_g":    int(rng.integers(t["carb"][0], t["carb"][1])),
            "fat_g":      int(rng.integers(t["fat"][0],  t["fat"][1])),
        })

nutrition_raw = pd.DataFrame(meal_rows)
print(f"  nutrition_raw shape: {nutrition_raw.shape}")

# ─── Also persist the workout schedule so load_data.py can build Dim_Workout ─

workout_schedule = pd.DataFrame({
    "full_date":       DATE_STRS,
    "workout_type":    WORKOUT_TYPES,
    "exercise_category": CATEGORY,
    "intensity":       INTENSITY,
    "duration_minutes": DURATION,
})

# ─── persist raw sources ──────────────────────────────────────────────────────

out_dir = os.path.dirname(os.path.abspath(__file__))
wearable_raw.to_csv(  os.path.join(out_dir, "wearable_raw.csv"),   index=False)
nutrition_raw.to_csv( os.path.join(out_dir, "nutrition_raw.csv"),  index=False)
workout_schedule.to_csv(os.path.join(out_dir, "workout_schedule.csv"), index=False)

print("\n✓ Raw sources written")
print(f"  wearable_raw.csv   : {len(wearable_raw):,} rows")
print(f"  nutrition_raw.csv  : {len(nutrition_raw):,} rows")
print(f"  workout_schedule.csv: {len(workout_schedule):,} rows")
