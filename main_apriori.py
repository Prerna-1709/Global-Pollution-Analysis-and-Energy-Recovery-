"""Orchestration script for Association Rule Mining (Step 4).

Run:
    python main_apriori.py
"""

import os
import numpy as np
import pandas as pd
from src.preprocessing import DataPreprocessor
from src.models.association_rules import AssociationRuleMiner


def main():
    csv_path = 'data/Global_Pollution_Analysis.csv'

    # ── Create dummy dataset if the real CSV is absent ──────────────────────
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Generating a dummy dataset.")
        os.makedirs('data', exist_ok=True)
        np.random.seed(0)
        n = 500
        df_dummy = pd.DataFrame({
            'Country': np.random.choice(['USA', 'China', 'India', 'Brazil', 'Germany'], n),
            'Year': np.random.choice([2019, 2020, 2021, 2022], n),
            'Air_Pollution_Index': np.random.uniform(15, 180, n),
            'Industrial_Waste': np.random.uniform(200, 6000, n),
            'CO2_Emissions': np.random.uniform(800, 12000, n),
            'GDP': np.random.uniform(3000, 55000, n),
            'Energy_Recovered_GWh': np.random.uniform(5, 600, n),
            'Energy_Consumption_Per_Capita': np.random.uniform(500, 15000, n),
            'Recovery_Rate_percent': np.random.uniform(5, 95, n),
        })
        df_dummy.to_csv(csv_path, index=False)

    # ── Load & impute missing values ─────────────────────────────────────────
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(csv_path)
    df = preprocessor.handle_missing_values(df, strategy='mean')

    # ── Define columns to discretize ─────────────────────────────────────────
    continuous_cols = []
    candidates = [
        'Air_Pollution_Index', 'Pollution_Index',
        'CO2_Emissions', 'Industrial_Waste',
        'Energy_Recovered_GWh', 'Energy_Consumption_Per_Capita',
        'Recovery_Rate_percent', 'GDP'
    ]
    for col in candidates:
        if col in df.columns:
            continuous_cols.append(col)

    if not continuous_cols:
        print("No suitable continuous columns found for discretization. Exiting.")
        return

    # ── Discretize ────────────────────────────────────────────────────────────
    miner = AssociationRuleMiner(min_support=0.08, min_confidence=0.4)
    df_disc = miner.discretize(df, columns=continuous_cols)

    level_cols = [f'{c}_Level' for c in continuous_cols if f'{c}_Level' in df_disc.columns]
    print(f"Discretized columns: {level_cols}")

    # ── Build transaction one-hot DataFrame ───────────────────────────────────
    transaction_df = miner.build_transaction_df(df_disc, item_columns=level_cols)
    print(f"Transaction matrix shape: {transaction_df.shape}")

    # ── Run Apriori ───────────────────────────────────────────────────────────
    rules = miner.fit(transaction_df)

    if rules.empty:
        print("No association rules generated. Try lowering min_support or min_confidence.")
        return

    # ── Print Top 10 rules by Lift ────────────────────────────────────────────
    top10 = miner.get_top_rules(10)
    print("\n=== Top 10 Association Rules (ranked by Lift) ===")
    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    print(top10[display_cols].to_string(index=False))

    # ── Insight: High Pollution → High Energy Recovery ────────────────────────
    print("\n=== Insight Rules: High Pollution -> High Energy Recovery ===")
    insight = miner.filter_insight_rules(keyword_antecedent='High', keyword_consequent='High')
    if insight.empty:
        print("No direct High->High insight rules found with current thresholds.")
    else:
        print(insight[display_cols].head(5).to_string(index=False))

    # ── Visualize network graph ───────────────────────────────────────────────
    miner.plot_rules_graph(top_n=10, save_path='outputs/association_rules_graph.png')


if __name__ == '__main__':
    main()
