"""Association Rule Mining module using Apriori algorithm.

This module discretizes continuous pollution/energy features into
categorical bins and mines frequent itemsets using mlxtend.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


class AssociationRuleMiner:
    """Mines association rules from discretized pollution/energy data.

    Attributes:
        min_support (float): Minimum support threshold for Apriori.
        min_confidence (float): Minimum confidence threshold for rule generation.
        rules (pd.DataFrame): Mined association rules after fitting.
    """

    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        """Initializes the miner with support and confidence thresholds.

        Args:
            min_support: Minimum support for Apriori algorithm.
            min_confidence: Minimum confidence for rule generation.
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = None
        self.frequent_itemsets = None

    def discretize(self, df: pd.DataFrame, columns: list, labels=None) -> pd.DataFrame:
        """Discretizes continuous columns into Low/Medium/High bins.

        Args:
            df: Input DataFrame with continuous features.
            columns: List of column names to discretize.
            labels: Bin labels. Defaults to ['Low', 'Medium', 'High'].

        Returns:
            DataFrame with discretized columns appended as '<col>_Level'.
        """
        if labels is None:
            labels = ['Low', 'Medium', 'High']
        df_out = df.copy()
        for col in columns:
            if col in df_out.columns:
                df_out[f'{col}_Level'] = pd.qcut(
                    df_out[col],
                    q=len(labels),
                    labels=[f'{col}_{l}' for l in labels],
                    duplicates='drop'
                )
        return df_out

    def build_transaction_df(self, df: pd.DataFrame, item_columns: list) -> pd.DataFrame:
        """Converts item columns into a one-hot encoded transaction DataFrame.

        Args:
            df: DataFrame containing discretized item columns.
            item_columns: Columns whose string values are treated as items.

        Returns:
            One-hot encoded boolean DataFrame suitable for mlxtend Apriori.
        """
        transactions = []
        for _, row in df.iterrows():
            basket = [str(row[col]) for col in item_columns if col in df.columns and pd.notna(row[col])]
            transactions.append(basket)

        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        return pd.DataFrame(te_array, columns=te.columns_)

    def fit(self, transaction_df: pd.DataFrame) -> pd.DataFrame:
        """Runs Apriori and generates association rules ranked by lift.

        Args:
            transaction_df: One-hot encoded boolean transaction DataFrame.

        Returns:
            DataFrame of association rules sorted by lift descending.
        """
        self.frequent_itemsets = apriori(
            transaction_df,
            min_support=self.min_support,
            use_colnames=True
        )

        if self.frequent_itemsets.empty:
            print("No frequent itemsets found. Try lowering min_support.")
            self.rules = pd.DataFrame()
            return self.rules

        self.rules = association_rules(
            self.frequent_itemsets,
            metric='confidence',
            min_threshold=self.min_confidence
        )
        self.rules = self.rules.sort_values('lift', ascending=False).reset_index(drop=True)
        return self.rules

    def get_top_rules(self, n: int = 10) -> pd.DataFrame:
        """Returns the top-n rules ranked by lift.

        Args:
            n: Number of top rules to return.

        Returns:
            Top-n rules DataFrame.
        """
        if self.rules is None or self.rules.empty:
            return pd.DataFrame()
        return self.rules.head(n)

    def filter_insight_rules(self, keyword_antecedent: str = 'High', keyword_consequent: str = 'High') -> pd.DataFrame:
        """Filters rules where antecedents and consequents contain specific keywords.

        Args:
            keyword_antecedent: Keyword to match in antecedent items.
            keyword_consequent: Keyword to match in consequent items.

        Returns:
            Filtered rules DataFrame.
        """
        if self.rules is None or self.rules.empty:
            return pd.DataFrame()

        mask = (
            self.rules['antecedents'].apply(lambda x: any(keyword_antecedent in item for item in x)) &
            self.rules['consequents'].apply(lambda x: any(keyword_consequent in item for item in x))
        )
        return self.rules[mask]

    def plot_rules_graph(self, top_n: int = 10, save_path: str = 'outputs/association_rules_graph.png'):
        """Generates a directed network graph of the top association rules.

        Args:
            top_n: Number of top rules to visualize.
            save_path: File path to save the graph image.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        top_rules = self.get_top_rules(top_n)
        if top_rules.empty:
            print("No rules to visualize.")
            return

        G = nx.DiGraph()
        for _, row in top_rules.iterrows():
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            lift = round(row['lift'], 2)
            G.add_edge(ant, con, weight=lift, label=f"lift={lift}")

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42, k=2)
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(weights) if weights else 1

        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='#4C72B0', alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=7, font_color='white', font_weight='bold')
        nx.draw_networkx_edges(
            G, pos,
            width=[1 + 3 * (w / max_w) for w in weights],
            edge_color='#DD8452',
            arrows=True,
            arrowsize=20,
            connectionstyle='arc3,rad=0.1'
        )
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

        plt.title('Association Rules Network Graph (Top Rules by Lift)', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Association rules graph saved to {save_path}")
