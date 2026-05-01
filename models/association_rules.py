"""
association_rules.py
--------------------
Apriori association rule mining on Dim_Nutrition × Dim_Workout data.

Discovers patterns such as:
  {Meal = Batch-cooked Poultry, Workout = Dumbbell Strength} => {Deep Sleep = High}

Outputs
-------
outputs/association_rules.csv
outputs/plots/arules_scatter.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
PLOTS_DIR  = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_SUPPORT    = 0.05
MIN_CONFIDENCE = 0.40
MIN_LIFT       = 1.1


def _build_transactions(df: pd.DataFrame) -> list:
    """
    Convert each daily record into a transaction (set of boolean items).
    Items encode: meal category, workout type, sleep quality, recovery state.
    """
    transactions = []
    for _, row in df.iterrows():
        items = []

        # Meal items
        items.append(f"Meal={row['meal_category']}")

        # Workout items
        items.append(f"Workout={row['workout_type']}")

        # Intensity bucket
        if row["intensity_score"] >= 2:
            items.append("Intensity=High")
        elif row["intensity_score"] == 1:
            items.append("Intensity=Low")
        else:
            items.append("Intensity=None")

        # Sleep quality
        if row["sleep_hours"] >= 7.5:
            items.append("Sleep=Deep")
        elif row["sleep_hours"] >= 6.0:
            items.append("Sleep=Adequate")
        else:
            items.append("Sleep=Poor")

        # High-protein flag
        if row["is_high_protein"]:
            items.append("HighProtein=Yes")
        else:
            items.append("HighProtein=No")

        # Recovery label
        items.append(f"Recovery={row['recovery_label']}")

        transactions.append(items)
    return transactions


def run_association_rules(df: pd.DataFrame) -> pd.DataFrame:
    transactions = _build_transactions(df)

    te      = TransactionEncoder()
    te_ary  = te.fit_transform(transactions)
    df_enc  = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = apriori(df_enc, min_support=MIN_SUPPORT, use_colnames=True)
    frequent.sort_values("support", ascending=False, inplace=True)
    print(f"[arules] Frequent itemsets found: {len(frequent)}")

    if len(frequent) == 0:
        print("[arules] No frequent itemsets found – lowering support threshold.")
        frequent = apriori(df_enc, min_support=0.02, use_colnames=True)

    rules = association_rules(frequent, metric="lift", min_threshold=MIN_LIFT)
    rules = rules[rules["confidence"] >= MIN_CONFIDENCE].sort_values(
        "lift", ascending=False
    )
    print(f"[arules] Rules found: {len(rules)}")

    if len(rules) > 0:
        print("\nTop 10 association rules:")
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print(rules[cols].head(10).to_string(index=False))

        # ── Scatter plot: support vs confidence, coloured by lift ─────────────
        fig, ax = plt.subplots(figsize=(9, 6))
        sc = ax.scatter(
            rules["support"], rules["confidence"],
            c=rules["lift"], cmap="YlOrRd", s=40, alpha=0.75, edgecolors="grey", linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax, label="Lift")
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_title("Association Rules – Support vs Confidence (colour = Lift)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "arules_scatter.png", dpi=150)
        plt.close()
        print("[arules] Saved arules_scatter.png")

    # serialise rules (convert frozensets to strings for CSV)
    rules_out = rules.copy()
    rules_out["antecedents"] = rules_out["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules_out["consequents"] = rules_out["consequents"].apply(lambda x: ", ".join(sorted(x)))
    out = OUTPUT_DIR / "association_rules.csv"
    rules_out.to_csv(out, index=False)
    print(f"[arules] Rules saved → {out}")
    return rules


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_DIR / "cleaned_data.csv")
    run_association_rules(df)
