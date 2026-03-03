"""Orchestration script for K-Means & Hierarchical Clustering (Step 5).

Run:
    python main_clustering.py
"""

import os
import numpy as np
import pandas as pd
from src.preprocessing import DataPreprocessor
from src.models.clustering import ClusterAnalyzer

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = 'data/Global_Pollution_Analysis.csv'
CLUSTER_FEATURES = ['Air_Pollution_Index', 'CO2_Emissions', 'Energy_Recovered_GWh']
POLLUTION_COL = 'Air_Pollution_Index'
RECOVERY_COL = 'Energy_Recovered_GWh'
COUNTRY_COL = 'Country'


def generate_dummy_data(path: str, n: int = 400):
    """Creates a synthetic dataset if the real CSV is missing."""
    np.random.seed(1)
    countries = ['USA', 'China', 'India', 'Brazil', 'Germany',
                 'Russia', 'Japan', 'Australia', 'Nigeria', 'Canada']
    df = pd.DataFrame({
        'Country': np.random.choice(countries, n),
        'Year': np.random.choice(range(2015, 2023), n),
        'Air_Pollution_Index': np.random.uniform(10, 200, n),
        'CO2_Emissions': np.random.uniform(500, 12000, n),
        'Energy_Recovered_GWh': np.random.uniform(5, 700, n),
        'GDP': np.random.uniform(3000, 60000, n),
        'Industrial_Waste': np.random.uniform(100, 7000, n),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dummy dataset created at {path} ({n} rows).")


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"Warning: {CSV_PATH} not found. Generating dummy dataset.")
        generate_dummy_data(CSV_PATH)

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(CSV_PATH)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    df = preprocessor.handle_missing_values(df, strategy='mean')

    # ── Select and validate clustering features ───────────────────────────────
    avail_features = [f for f in CLUSTER_FEATURES if f in df.columns]
    if len(avail_features) < 2:
        # Fallback: use any numeric columns
        avail_features = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
    print(f"Clustering features: {avail_features}")

    df_cluster = df[avail_features].copy()

    # ── Scale features (critical for distance-based clustering) ───────────────
    df_scaled = preprocessor.scale_features(df_cluster, columns=avail_features)
    X = df_scaled.values

    # ── Keep original unscaled for plotting + export ──────────────────────────
    df_orig = df.copy()

    analyzer = ClusterAnalyzer()

    # ── Elbow Method → optimal k ──────────────────────────────────────────────
    print("\n--- Elbow Method ---")
    optimal_k = analyzer.elbow_method(X, k_range=range(2, 11),
                                       save_path='outputs/elbow_plot.png')

    # ── K-Means ───────────────────────────────────────────────────────────────
    print(f"\n--- K-Means Clustering (k={optimal_k}) ---")
    kmeans_labels = analyzer.fit_kmeans(X, k=optimal_k)

    # ── Hierarchical Clustering ───────────────────────────────────────────────
    print(f"\n--- Hierarchical Clustering (n_clusters={optimal_k}) ---")
    hier_labels = analyzer.fit_hierarchical(X, n_clusters=optimal_k, linkage_method='ward')

    # ── Dendrogram ────────────────────────────────────────────────────────────
    country_labels = df_orig[COUNTRY_COL].tolist() if COUNTRY_COL in df_orig.columns else None
    analyzer.plot_dendrogram(labels=country_labels,
                             save_path='outputs/dendrogram.png',
                             truncate_p=30)

    # ── Scatter comparison plot ───────────────────────────────────────────────
    x_col = POLLUTION_COL if POLLUTION_COL in df_orig.columns else avail_features[0]
    y_col = RECOVERY_COL if RECOVERY_COL in df_orig.columns else avail_features[1]
    analyzer.plot_cluster_scatter(
        df_original=df_orig,
        x_col=x_col, y_col=y_col,
        kmeans_labels=kmeans_labels,
        hier_labels=hier_labels,
        save_path='outputs/cluster_scatter.png'
    )

    # ── Export cluster results CSV ────────────────────────────────────────────
    df_results = analyzer.export_results(
        df_original=df_orig,
        country_col=COUNTRY_COL,
        kmeans_labels=kmeans_labels,
        hier_labels=hier_labels,
        save_path='outputs/cluster_results.csv'
    )

    # ── At-risk country identification ────────────────────────────────────────
    if POLLUTION_COL in df_orig.columns and RECOVERY_COL in df_orig.columns:
        print("\n--- At-Risk Countries (High Pollution / Low Recovery) ---")
        at_risk = analyzer.identify_at_risk(
            df_results=df_results,
            df_original=df_orig,
            pollution_col=POLLUTION_COL,
            recovery_col=RECOVERY_COL,
            country_col=COUNTRY_COL
        )
        print(at_risk[[COUNTRY_COL, 'KMeans_Cluster']].value_counts(COUNTRY_COL).head(10).to_string())

    print("\nDone. Check the outputs/ folder for all generated files.")


if __name__ == '__main__':
    main()
