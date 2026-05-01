"""
main.py
-------
End-to-end orchestration script for the Bimodal Biometric Warehouse project.

Run this single script to reproduce the full pipeline:
    python main.py

Steps executed
--------------
1. Generate synthetic wearable + nutrition data (365 days)
2. Build SQLite star-schema database
3. Preprocess: clean, impute, feature-engineer
4. PCA on heart-rate feature space
5. EDA: histograms, heatmaps, scatter plots
6. K-Means clustering
7. Apriori association rule mining
8. Linear Regression (calorie burn prediction)
9. Random Forest + SVM classification (recovery state)
10. Consolidated evaluation report
"""

import sys
from pathlib import Path

# ── ensure sub-packages are importable ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_data      import main as gen_data
from database.setup_db       import main as setup_db
from analysis.preprocessing  import run_preprocessing
from analysis.pca_analysis   import run_pca
from analysis.eda             import run_eda
from models.clustering        import run_clustering
from models.association_rules import run_association_rules
from models.regression        import run_regression
from models.classification    import run_classification
from evaluation.evaluate      import run_evaluation

import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "outputs"


def banner(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    banner("Step 1 – Generate Synthetic Data")
    gen_data()

    banner("Step 2 – Build Star-Schema Database")
    setup_db()

    banner("Step 3 – Preprocessing & Feature Engineering")
    df = run_preprocessing()

    banner("Step 4 – PCA on Heart-Rate Feature Space")
    run_pca(df)

    banner("Step 5 – Exploratory Data Analysis")
    run_eda(df)

    banner("Step 6 – K-Means Clustering")
    run_clustering(df)

    banner("Step 7 – Apriori Association Rule Mining")
    run_association_rules(df)

    banner("Step 8a – Linear Regression (Calorie Prediction)")
    run_regression(df)

    banner("Step 8b – Classification (Recovery Prediction)")
    run_classification(df)

    banner("Step 9 – Consolidated Model Evaluation")
    run_evaluation(df)

    banner("Pipeline Complete")
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    print("To launch the prediction API:  python app/flask_app.py")


if __name__ == "__main__":
    main()
