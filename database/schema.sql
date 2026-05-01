-- schema.sql
-- Star-schema DDL for the Bimodal Biometric Warehouse
-- Compatible with SQLite, MySQL, and PostgreSQL
-- (SQLite ignores column-type widths and UNSIGNED; adapt for production)

-- ─────────────────────────────────────────────────────────────────────────────
-- Dimension Tables
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS Dim_Time (
    time_id     INTEGER PRIMARY KEY,
    date        TEXT    NOT NULL,       -- ISO-8601 YYYY-MM-DD
    day_of_week TEXT    NOT NULL,
    month       INTEGER NOT NULL,
    season      TEXT    NOT NULL,       -- Winter / Spring / Summer / Fall
    is_weekend  INTEGER NOT NULL        -- 0 = weekday, 1 = weekend
);

CREATE TABLE IF NOT EXISTS Dim_Workout (
    workout_id        INTEGER PRIMARY KEY,
    workout_type      TEXT    NOT NULL,  -- e.g. "Dumbbell Strength"
    exercise_category TEXT    NOT NULL,  -- Strength / Cardio / Rest / Recovery
    intensity         INTEGER NOT NULL,  -- 1–10 scale
    duration_minutes  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS Dim_Nutrition (
    nutrition_id     INTEGER PRIMARY KEY,
    meal_category    TEXT    NOT NULL,   -- e.g. "Batch-cooked Poultry"
    total_calories   INTEGER NOT NULL,
    protein_g        INTEGER NOT NULL,
    carbs_g          INTEGER NOT NULL,
    fat_g            INTEGER NOT NULL,
    is_high_protein  INTEGER NOT NULL,   -- 0 / 1 flag
    is_poultry       INTEGER NOT NULL,
    is_vegetarian    INTEGER NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Fact Table
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS Fact_Daily_Biometrics (
    fact_id                INTEGER PRIMARY KEY,
    date_id                INTEGER NOT NULL REFERENCES Dim_Time(time_id),
    workout_id             INTEGER NOT NULL REFERENCES Dim_Workout(workout_id),
    nutrition_id           INTEGER NOT NULL REFERENCES Dim_Nutrition(nutrition_id),
    time_id                INTEGER NOT NULL REFERENCES Dim_Time(time_id),
    total_active_minutes   INTEGER NOT NULL,
    resting_heart_rate     INTEGER NOT NULL,   -- bpm
    sleep_duration_hours   REAL    NOT NULL,
    active_calories        INTEGER NOT NULL,
    steps                  INTEGER NOT NULL,
    hrv_score              INTEGER NOT NULL,   -- 0–100
    recovery_score         INTEGER NOT NULL,   -- 0–100 composite
    recovery_label         INTEGER NOT NULL    -- 1=Ready to Train, 0=Needs Rest
);
