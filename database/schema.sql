-- schema.sql
-- Star-schema DDL for the Bimodal Biometric Warehouse
-- Compatible with SQLite, MySQL, and PostgreSQL
-- Surrogate keys (integer auto-increment) are used throughout so the schema
-- supports multiple workout sessions per day without natural-key collisions.

-- ─────────────────────────────────────────────────────────────────────────────
-- Dimension Tables
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS Dim_Date (
    date_sk     INTEGER PRIMARY KEY,          -- surrogate key
    full_date   TEXT    NOT NULL UNIQUE,      -- natural key (ISO YYYY-MM-DD)
    day_of_week TEXT    NOT NULL,
    month       INTEGER NOT NULL,
    quarter     INTEGER NOT NULL,
    season      TEXT    NOT NULL,             -- Winter / Spring / Summer / Fall
    is_weekend  INTEGER NOT NULL              -- 0 = weekday, 1 = weekend
);

CREATE TABLE IF NOT EXISTS Dim_Workout (
    workout_sk        INTEGER PRIMARY KEY,    -- surrogate key
    full_date         TEXT    NOT NULL,       -- natural FK → Dim_Date.full_date
    workout_type      TEXT    NOT NULL,       -- e.g. "Dumbbell Strength"
    exercise_category TEXT    NOT NULL,       -- Strength / Cardio / Rest / Recovery
    intensity         INTEGER NOT NULL,       -- 1–10 scale
    duration_minutes  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS Dim_Nutrition (
    nutrition_sk     INTEGER PRIMARY KEY,     -- surrogate key
    full_date        TEXT    NOT NULL,        -- natural FK → Dim_Date.full_date
    meal_category    TEXT    NOT NULL,        -- derived dominant category
    total_calories   INTEGER NOT NULL,
    protein_g        REAL    NOT NULL,
    carbs_g          REAL    NOT NULL,
    fat_g            REAL    NOT NULL,
    is_high_protein  INTEGER NOT NULL,        -- 1 if protein_g > 150
    is_poultry       INTEGER NOT NULL,        -- 1 if dominant meal is poultry
    is_vegetarian    INTEGER NOT NULL         -- 1 if dominant meal is vegetarian
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Fact Table
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS Fact_Daily_Biometrics (
    fact_sk                INTEGER PRIMARY KEY,
    date_sk                INTEGER NOT NULL REFERENCES Dim_Date(date_sk),
    workout_sk             INTEGER NOT NULL REFERENCES Dim_Workout(workout_sk),
    nutrition_sk           INTEGER NOT NULL REFERENCES Dim_Nutrition(nutrition_sk),
    -- continuous measures
    total_active_minutes   INTEGER NOT NULL,
    resting_heart_rate     REAL    NOT NULL,  -- bpm (daily min of minute-level HR)
    sleep_duration_hours   REAL    NOT NULL,
    active_calories        INTEGER NOT NULL,
    steps                  INTEGER NOT NULL,
    hrv_score              REAL    NOT NULL,  -- 0–100 composite
    -- PCA-derived heart-rate components (computed in notebook, stored for reference)
    hr_pc1                 REAL,
    hr_pc2                 REAL,
    hr_pc3                 REAL,
    hr_pc4                 REAL,
    hr_pc5                 REAL,
    -- lag features (t-1 day, computed during notebook preprocessing)
    lag1_sleep             REAL,
    lag1_active_calories   REAL,
    lag1_hrv               REAL,
    -- targets
    recovery_score         REAL    NOT NULL,  -- 0–100 composite score
    recovery_label         INTEGER NOT NULL   -- 1=Ready to Train, 0=Needs Rest
);
