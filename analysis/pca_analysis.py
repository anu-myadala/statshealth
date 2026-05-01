"""
pca_analysis.py
---------------
Applies PCA to minute-by-minute heart rate proxies (simulated from daily metrics)
to reduce dimensionality and extract key feature vectors.

Outputs
-------
outputs/pca_components.csv   – principal component scores per day
outputs/plots/pca_variance.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
PLOTS_DIR  = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def simulate_hr_features(df: pd.DataFrame, n_features: int = 20) -> np.ndarray:
    """
    Simulate a set of heart-rate–derived features per day.

    In a real project these would be minute-by-minute HR readings.  Here we
    generate 20 correlated features (zones, variability, percentiles, etc.)
    from the available scalar metrics so that PCA has meaningful structure to
    extract.
    """
    rng = np.random.default_rng(SEED)
    n   = len(df)

    rhr   = df["resting_hr"].values
    act   = df["active_minutes"].values
    inten = df["intensity_score"].values
    sleep = df["sleep_hours"].values

    base = np.column_stack([
        rhr,                                         # 0 resting HR
        rhr + inten * 10 + rng.normal(0, 3, n),      # 1 mean active HR
        rhr + inten * 20 + rng.normal(0, 5, n),      # 2 peak HR
        rhr - sleep * 0.8 + rng.normal(0, 2, n),     # 3 overnight min HR
        act * 0.5 + rng.normal(0, 4, n),             # 4 time in zone 2
        act * 0.3 * inten + rng.normal(0, 3, n),     # 5 time in zone 3-4
        inten * 5 + rng.normal(0, 2, n),             # 6 HR zone distribution
        rhr + rng.normal(0, 2, n),                   # 7 HR at rest start
        rhr - 2 + rng.normal(0, 1, n),               # 8 HR at rest end
        np.clip(100 - rhr * 1.1, 20, 80) + rng.normal(0, 3, n),  # 9 HRV proxy
    ])

    # add 10 more noisy correlated features
    noise = rng.normal(0, 1, (n, 10))
    extra = base[:, :10] @ rng.normal(0, 0.3, (10, 10)) + noise
    return np.hstack([base, extra])


def run_pca(df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    X    = simulate_hr_features(df)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pca   = PCA(n_components=n_components, random_state=SEED)
    comps = pca.fit_transform(X_sc)

    # ── Explained-variance plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_ * 100,
        color="steelblue", edgecolor="white",
    )
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("PCA – Explained Variance per Component")

    axes[1].plot(
        range(1, n_components + 1),
        np.cumsum(pca.explained_variance_ratio_) * 100,
        marker="o", color="darkorange",
    )
    axes[1].axhline(90, linestyle="--", color="gray", alpha=0.6, label="90 % threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("PCA – Cumulative Explained Variance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pca_variance.png", dpi=150)
    plt.close()
    print("[pca] Saved pca_variance.png")

    # ── Return component scores ──────────────────────────────────────────────
    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    result  = pd.DataFrame(comps, columns=pc_cols)
    result.insert(0, "date", df["date"].values)

    for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i}: {ratio * 100:.1f}% variance explained")

    out = OUTPUT_DIR / "pca_components.csv"
    result.to_csv(out, index=False)
    print(f"[pca] PCA components saved → {out}")
    return result


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_DIR / "cleaned_data.csv")
    run_pca(df)
