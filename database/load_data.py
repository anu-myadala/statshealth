"""
load_data.py
Reads the four CSVs produced by generate_data.py and loads them into a
local SQLite database (biometric_warehouse.db).

Usage:
    python database/load_data.py
"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(BASE_DIR, "database", "biometric_warehouse.db")
SQL_PATH = os.path.join(BASE_DIR, "database", "schema.sql")


def load():
    # ── create schema ─────────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    with open(SQL_PATH) as f:
        conn.executescript(f.read())
    conn.commit()

    # ── load dimension tables first (FK order) ─────────────────────────────────
    tables = {
        "Dim_Time":              "dim_time.csv",
        "Dim_Workout":           "dim_workout.csv",
        "Dim_Nutrition":         "dim_nutrition.csv",
        "Fact_Daily_Biometrics": "fact_daily_biometrics.csv",
    }

    for table, csv_file in tables.items():
        csv_path = os.path.join(DATA_DIR, csv_file)
        df = pd.read_csv(csv_path)
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"  ✓ Loaded {len(df):>4} rows → {table}")

    conn.close()
    print(f"\n✓ Database written to {DB_PATH}")


if __name__ == "__main__":
    load()
