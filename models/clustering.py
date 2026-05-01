"""
clustering.py
-------------
K-Means clustering on daily biometric logs.

Goal: Group days into distinct lifestyle clusters
      ("High Strain / Low Recovery", "Optimal Balance", "Sedentary")

Outputs
-------
outputs/cluster_labels.csv
outputs/plots/kmeans_clusters.png
outputs/plots/kmeans_elbow.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
PLOTS_DIR  = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLUSTERS = 3

CLUSTER_FEATURES = [
    "steps", "active_minutes", "resting_hr", "sleep_hours",
    "calories_burned", "intensity_score", "protein_g", "recovery_score",
]

CLUSTER_NAMES = {
    0: "High Strain / Low Recovery",
    1: "Optimal Balance",
    2: "Sedentary",
}


def _rename_clusters(df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """
    Map numeric cluster IDs to descriptive names based on mean active_minutes.
    Highest active → "High Strain", middle → "Optimal Balance", lowest → "Sedentary".
    """
    cluster_ids  = sorted(set(labels))
    means        = {c: df.loc[labels == c, "active_minutes"].mean() for c in cluster_ids}
    sorted_ids   = sorted(means, key=means.get, reverse=True)
    name_map     = {
        sorted_ids[0]: "High Strain / Low Recovery",
        sorted_ids[1]: "Optimal Balance",
        sorted_ids[2]: "Sedentary",
    }
    return np.array([name_map[l] for l in labels])


def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    X       = df[CLUSTER_FEATURES].copy().fillna(df[CLUSTER_FEATURES].mean())
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    # ── Elbow method ──────────────────────────────────────────────────────────
    inertias = []
    k_range  = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(X_sc)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_range, inertias, marker="o", color="steelblue")
    ax.axvline(N_CLUSTERS, linestyle="--", color="red", alpha=0.6, label=f"Selected k={N_CLUSTERS}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("K-Means Elbow Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kmeans_elbow.png", dpi=150)
    plt.close()
    print("[clustering] Saved kmeans_elbow.png")

    # ── Fit final model ───────────────────────────────────────────────────────
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_sc)

    named_labels = _rename_clusters(df, labels)
    df           = df.copy()
    df["cluster_id"]   = labels
    df["cluster_name"] = named_labels

    # ── Visualise in 2-D PCA space ────────────────────────────────────────────
    pca  = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_sc)

    palette = {
        "High Strain / Low Recovery": "crimson",
        "Optimal Balance":            "seagreen",
        "Sedentary":                  "steelblue",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, grp_idx in [(n, named_labels == n) for n in palette]:
        ax.scatter(X_2d[grp_idx, 0], X_2d[grp_idx, 1],
                   label=name, color=palette[name], alpha=0.6, s=25)

    # plot centroids projected to 2-D
    centroids_2d = pca.transform(km.cluster_centers_)
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
               marker="X", s=200, color="black", zorder=5, label="Centroid")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("K-Means Clusters of Daily Activity Profiles (PCA Projection)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kmeans_clusters.png", dpi=150)
    plt.close()
    print("[clustering] Saved kmeans_clusters.png")

    # ── Summary statistics per cluster ────────────────────────────────────────
    summary = df.groupby("cluster_name")[CLUSTER_FEATURES].mean().round(2)
    print("\n[clustering] Cluster centroids (feature means):")
    print(summary.to_string())

    out = OUTPUT_DIR / "cluster_labels.csv"
    df[["date", "cluster_id", "cluster_name"]].to_csv(out, index=False)
    print(f"\n[clustering] Labels saved → {out}")
    return df


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_DIR / "cleaned_data.csv")
    run_clustering(df)
