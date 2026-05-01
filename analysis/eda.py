"""
eda.py
------
Exploratory Data Analysis and Visualization.

Outputs (in outputs/plots/)
---------------------------
eda_distributions.png   – histograms of key metrics
eda_correlation_hmap.png – full correlation heat-map
eda_protein_vs_rhr.png   – high-protein intake vs next-day RHR
eda_workout_sleep.png    – sleep by workout type (box plots)
eda_weekly_pattern.png   – mean calories burned by day of week
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
PLOTS_DIR  = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Histograms
# ──────────────────────────────────────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame) -> None:
    metrics = ["steps", "active_minutes", "resting_hr", "sleep_hours",
               "calories_burned", "protein_g", "total_calories"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(metrics):
        axes[i].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    axes[-1].axis("off")  # hide unused subplot
    plt.suptitle("Distribution of Key Biometric & Nutritional Metrics", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[eda] Saved eda_distributions.png")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Correlation heat-map
# ──────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_cols = [
        "steps", "active_minutes", "resting_hr", "sleep_hours",
        "calories_burned", "total_calories", "protein_g", "carbs_g",
        "fat_g", "intensity_score", "recovery_score",
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title("Correlation Heat-Map – Biometric & Nutritional Features", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_correlation_hmap.png", dpi=150)
    plt.close()
    print("[eda] Saved eda_correlation_hmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# 3. High-protein intake vs next-day resting HR
# ──────────────────────────────────────────────────────────────────────────────
def plot_protein_vs_rhr(df: pd.DataFrame) -> None:
    tmp = df[["protein_g", "next_day_rhr", "is_high_protein"]].dropna()
    tmp["Protein Level"] = tmp["is_high_protein"].map({1: "High Protein (≥150 g)", 0: "Normal Protein"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # scatter
    colors = {"High Protein (≥150 g)": "darkorange", "Normal Protein": "steelblue"}
    for label, grp in tmp.groupby("Protein Level"):
        axes[0].scatter(grp["protein_g"], grp["next_day_rhr"],
                        alpha=0.45, s=20, color=colors[label], label=label)
    axes[0].set_xlabel("Protein Intake (g)")
    axes[0].set_ylabel("Next-Day Resting HR (bpm)")
    axes[0].set_title("Protein Intake vs Next-Day Resting HR")
    axes[0].legend()

    # box plot
    sns.boxplot(data=tmp, x="Protein Level", y="next_day_rhr",
                hue="Protein Level", legend=False,
                palette={"High Protein (≥150 g)": "darkorange", "Normal Protein": "steelblue"},
                ax=axes[1])
    axes[1].set_title("Next-Day Resting HR by Protein Level")
    axes[1].set_ylabel("Next-Day Resting HR (bpm)")
    axes[1].set_xlabel("")

    plt.suptitle("High-Protein Intake Days Correlated with Next-Day Resting Heart Rate",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_protein_vs_rhr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[eda] Saved eda_protein_vs_rhr.png")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Sleep duration by workout type
# ──────────────────────────────────────────────────────────────────────────────
def plot_sleep_by_workout(df: pd.DataFrame) -> None:
    order = df.groupby("workout_type")["sleep_hours"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="workout_type", y="sleep_hours", order=order,
                hue="workout_type", legend=False,
                palette="Set2", ax=ax)
    ax.set_title("Sleep Duration by Workout Type")
    ax.set_xlabel("Workout Type")
    ax.set_ylabel("Sleep Hours")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_workout_sleep.png", dpi=150)
    plt.close()
    print("[eda] Saved eda_workout_sleep.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Weekly calorie-burn pattern
# ──────────────────────────────────────────────────────────────────────────────
def plot_weekly_pattern(df: pd.DataFrame) -> None:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly = df.groupby("day_of_week")["calories_burned"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(10, 4))
    weekly.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Mean Calories Burned by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Avg Calories Burned")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_weekly_pattern.png", dpi=150)
    plt.close()
    print("[eda] Saved eda_weekly_pattern.png")


def run_eda(df: pd.DataFrame) -> None:
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_protein_vs_rhr(df)
    plot_sleep_by_workout(df)
    plot_weekly_pattern(df)
    print("[eda] All EDA plots generated.")


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_DIR / "cleaned_data.csv")
    run_eda(df)
