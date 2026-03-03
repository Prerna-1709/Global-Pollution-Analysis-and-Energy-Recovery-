"""Clustering module for country environmental segmentation.

Implements K-Means (with Elbow Method) and Agglomerative Hierarchical
Clustering, plus comparison scatter plots and dendrogram visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


class ClusterAnalyzer:
    """Segments countries using K-Means and Hierarchical Clustering.

    Attributes:
        kmeans_model: Fitted KMeans instance.
        hier_model: Fitted AgglomerativeClustering instance.
        kmeans_labels (np.ndarray): Cluster labels from K-Means.
        hier_labels (np.ndarray): Cluster labels from Hierarchical clustering.
    """

    def __init__(self):
        self.kmeans_model = None
        self.hier_model = None
        self.kmeans_labels = None
        self.hier_labels = None
        self._linkage_matrix = None

    # ── K-Means ────────────────────────────────────────────────────────────────

    def elbow_method(self, X: np.ndarray, k_range: range = range(2, 11),
                     save_path: str = 'outputs/elbow_plot.png') -> int:
        """Runs the Elbow Method and saves the inertia plot.

        Args:
            X: Scaled feature matrix.
            k_range: Range of k values to test.
            save_path: Path to save the elbow plot PNG.

        Returns:
            Optimal k suggested by the largest drop in inertia.
        """
        inertias = []
        silhouettes = []
        ks = list(k_range)

        silhouette_ks = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            if k > 1:
                silhouettes.append(silhouette_score(X, km.labels_))
                silhouette_ks.append(k)

        # Plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()
        ax1.plot(ks, inertias, 'o-', color='#4C72B0', label='Inertia')
        ax2.plot(silhouette_ks, silhouettes, 's--', color='#DD8452', label='Silhouette')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia', color='#4C72B0')
        ax2.set_ylabel('Silhouette Score', color='#DD8452')
        plt.title('Elbow Method & Silhouette Scores')
        fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Elbow plot saved to {save_path}")

        # Suggest optimal k: largest silhouette
        optimal_k = ks[1 + int(np.argmax(silhouettes))]
        print(f"Suggested optimal k (best silhouette): {optimal_k}")
        return optimal_k

    def fit_kmeans(self, X: np.ndarray, k: int) -> np.ndarray:
        """Fits K-Means with the specified k.

        Args:
            X: Scaled feature matrix.
            k: Number of clusters.

        Returns:
            Cluster label array.
        """
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.kmeans_labels = self.kmeans_model.fit_predict(X)
        score = silhouette_score(X, self.kmeans_labels)
        print(f"K-Means (k={k}) Silhouette Score: {score:.4f}")
        return self.kmeans_labels

    # ── Hierarchical Clustering ────────────────────────────────────────────────

    def fit_hierarchical(self, X: np.ndarray, n_clusters: int,
                         linkage_method: str = 'ward') -> np.ndarray:
        """Fits Agglomerative Hierarchical Clustering.

        Args:
            X: Scaled feature matrix.
            n_clusters: Number of clusters to cut the dendrogram at.
            linkage_method: Linkage criterion ('ward', 'complete', etc.).

        Returns:
            Cluster label array.
        """
        self.hier_model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage_method
        )
        self.hier_labels = self.hier_model.fit_predict(X)
        score = silhouette_score(X, self.hier_labels)
        print(f"Hierarchical (n={n_clusters}, linkage={linkage_method}) Silhouette Score: {score:.4f}")
        # Compute linkage matrix for dendrogram
        self._linkage_matrix = linkage(X, method=linkage_method)
        return self.hier_labels

    def plot_dendrogram(self, labels: list = None,
                        save_path: str = 'outputs/dendrogram.png',
                        truncate_p: int = 30):
        """Saves a dendrogram of the hierarchical clustering.

        Args:
            labels: Row labels (e.g., country names).
            save_path: Path to save the dendrogram PNG.
            truncate_p: Show only the last p merged clusters (readability).
        """
        if self._linkage_matrix is None:
            print("Fit hierarchical clustering first.")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(14, 6))
        dendrogram(
            self._linkage_matrix,
            labels=labels,
            truncate_mode='lastp',
            p=truncate_p,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=0.7 * max(self._linkage_matrix[:, 2])
        )
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample / Cluster')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Dendrogram saved to {save_path}")

    # ── Comparison & visualization ─────────────────────────────────────────────

    def plot_cluster_scatter(self, df_original: pd.DataFrame,
                             x_col: str, y_col: str,
                             kmeans_labels: np.ndarray,
                             hier_labels: np.ndarray,
                             save_path: str = 'outputs/cluster_scatter.png'):
        """Side-by-side scatter plots coloured by K-Means and Hierarchical labels.

        Args:
            df_original: DataFrame with original (unscaled) feature values.
            x_col: Column name for x-axis (e.g., 'Air_Pollution_Index').
            y_col: Column name for y-axis (e.g., 'Energy_Recovered_GWh').
            kmeans_labels: K-Means cluster assignments.
            hier_labels: Hierarchical cluster assignments.
            save_path: Path to save the PNG.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, labels, title in zip(
            axes,
            [kmeans_labels, hier_labels],
            ['K-Means Clusters', 'Hierarchical Clusters']
        ):
            scatter = ax.scatter(
                df_original[x_col], df_original[y_col],
                c=labels, cmap='tab10', alpha=0.7, edgecolors='k', linewidths=0.3
            )
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            plt.colorbar(scatter, ax=ax, label='Cluster ID')

        plt.suptitle('Pollution vs Energy Recovery by Cluster', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Cluster scatter plot saved to {save_path}")

    # ── Results export ─────────────────────────────────────────────────────────

    def export_results(self, df_original: pd.DataFrame,
                       country_col: str,
                       kmeans_labels: np.ndarray,
                       hier_labels: np.ndarray,
                       save_path: str = 'outputs/cluster_results.csv') -> pd.DataFrame:
        """Saves a CSV mapping each country to its K-Means and Hierarchical cluster.

        Args:
            df_original: Original DataFrame (pre-scaling).
            country_col: Column name holding country identifiers.
            kmeans_labels: K-Means cluster assignments.
            hier_labels: Hierarchical cluster assignments.
            save_path: Path to save the CSV.

        Returns:
            Results DataFrame.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result = df_original[[country_col]].copy() if country_col in df_original.columns else pd.DataFrame()
        result['KMeans_Cluster'] = kmeans_labels
        result['Hierarchical_Cluster'] = hier_labels
        result.to_csv(save_path, index=False)
        print(f"Cluster results saved to {save_path}")
        return result

    def identify_at_risk(self, df_results: pd.DataFrame,
                         df_original: pd.DataFrame,
                         pollution_col: str,
                         recovery_col: str,
                         country_col: str = 'Country') -> pd.DataFrame:
        """Identifies the 'at-risk' cluster: highest pollution, lowest recovery.

        Args:
            df_results: DataFrame with KMeans_Cluster column.
            df_original: Original feature DataFrame.
            pollution_col: Column representing pollution level.
            recovery_col: Column representing energy/recovery level.
            country_col: Column for country names.

        Returns:
            DataFrame of at-risk countries.
        """
        combined = pd.concat(
            [df_results.reset_index(drop=True), df_original[[pollution_col, recovery_col]].reset_index(drop=True)],
            axis=1
        )
        cluster_stats = combined.groupby('KMeans_Cluster').agg(
            avg_pollution=(pollution_col, 'mean'),
            avg_recovery=(recovery_col, 'mean')
        )
        # At-risk: high pollution, low recovery → highest pollution / lowest recovery ratio
        cluster_stats['risk_score'] = cluster_stats['avg_pollution'] / (cluster_stats['avg_recovery'] + 1e-9)
        at_risk_cluster = cluster_stats['risk_score'].idxmax()
        print(f"\nAt-risk cluster ID (K-Means): {at_risk_cluster}")
        print(cluster_stats)
        at_risk_countries = combined[combined['KMeans_Cluster'] == at_risk_cluster]
        return at_risk_countries
