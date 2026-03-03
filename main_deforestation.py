"""Orchestration script for SVM Deforestation Analysis (Step 3).

Configuration flag:
    TARGET_COLUMN  -- toggle between 'Forest_Loss_Area_km2' (default)
                      or 'Tree_Cover_Loss_percent'

Run:
    python main_deforestation.py
"""

import os
import numpy as np
import pandas as pd
from src.preprocessing import DataPreprocessor
from src.models.svm_analyzer import SVMAnalyzer

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_COLUMN = 'Forest_Loss_Area_km2'   # Toggle: 'Forest_Loss_Area_km2' | 'Tree_Cover_Loss_percent'
CSV_PATH = 'data/deforestation_dataset.csv'


def generate_dummy_dataset(path: str, n: int = 400):
    """Creates a synthetic deforestation dataset when the real CSV is absent."""
    np.random.seed(7)
    df = pd.DataFrame({
        'Country': np.random.choice(['Brazil', 'Indonesia', 'Congo', 'India', 'Mexico'], n),
        'Year': np.random.choice(range(2010, 2023), n),
        'Rainfall_mm': np.random.uniform(500, 3000, n),
        'GDP_Billion_USD': np.random.uniform(50, 2000, n),
        'Population': np.random.randint(5_000_000, 1_400_000_000, n),
        'Temperature_C': np.random.uniform(15, 35, n),
        'Agricultural_Expansion_km2': np.random.uniform(100, 50000, n),
        'Forest_Loss_Area_km2': np.random.uniform(50, 20000, n),
        'Tree_Cover_Loss_percent': np.random.uniform(0.5, 40.0, n),
    })
    # Introduce a few missing values
    df.loc[np.random.choice(n, 20, replace=False), 'Rainfall_mm'] = np.nan
    df.loc[np.random.choice(n, 15, replace=False), 'GDP_Billion_USD'] = np.nan
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dummy dataset created at {path} with {n} samples.")


def main():
    # ── Data loading ──────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"Warning: {CSV_PATH} not found. Generating dummy dataset.")
        generate_dummy_dataset(CSV_PATH)

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(CSV_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Missing value imputation ───────────────────────────────────────────────
    df = preprocessor.handle_missing_values(df, strategy='mean')

    # ── Encode categoricals ───────────────────────────────────────────────────
    cat_cols = [c for c in ['Country', 'Year'] if c in df.columns]
    df = preprocessor.encode_features(df, columns=cat_cols)

    # ── Validate target ───────────────────────────────────────────────────────
    if TARGET_COLUMN not in df.columns:
        fallback = 'Tree_Cover_Loss_percent' if TARGET_COLUMN == 'Forest_Loss_Area_km2' else 'Forest_Loss_Area_km2'
        print(f"Warning: '{TARGET_COLUMN}' not found. Falling back to '{fallback}'.")
        target = fallback
    else:
        target = TARGET_COLUMN

    # Drop the other target to avoid leakage
    other_target = 'Tree_Cover_Loss_percent' if target == 'Forest_Loss_Area_km2' else 'Forest_Loss_Area_km2'
    if other_target in df.columns:
        df = df.drop(columns=[other_target])

    # ── Feature scaling ───────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [c for c in num_cols if c != target]
    df = preprocessor.scale_features(df, columns=cols_to_scale)

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = preprocessor.split_data(df, target_column=target, test_size=0.2)
    feature_names = X_train.columns.tolist()
    X_train_arr, X_test_arr = X_train.values, X_test.values
    y_train_arr, y_test_arr = y_train.values, y_test.values

    print(f"\nTarget: '{target}'")
    print(f"Train shape: {X_train_arr.shape}  |  Test shape: {X_test_arr.shape}")

    # ── SVM + GridSearchCV ────────────────────────────────────────────────────
    print("\n--- SVM GridSearchCV (kernels: linear/rbf/poly, 5-fold CV) ---")
    analyzer = SVMAnalyzer(cv=5)
    analyzer.fit(X_train_arr, y_train_arr, feature_names=feature_names)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n--- Evaluation Metrics ---")
    metrics = analyzer.evaluate(X_test_arr, y_test_arr)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Cross-validation on full data ─────────────────────────────────────────
    print("\n--- 5-Fold Cross-Validation (best model) ---")
    X_all = df.drop(columns=[target]).values
    y_all = df[target].values
    analyzer.cross_validate(X_all, y_all)

    # ── Feature Importance ────────────────────────────────────────────────────
    print("\n--- Feature Importance (Permutation) ---")
    importances = analyzer.compute_feature_importance(X_test_arr, y_test_arr)
    print(importances.to_string())

    # ── Visualizations ────────────────────────────────────────────────────────
    analyzer.plot_feature_importance(save_path='outputs/feature_importance.png')
    analyzer.plot_top_feature_scatter(X_test_arr, y_test_arr, save_path='outputs/top_feature_scatter.png')


if __name__ == '__main__':
    main()
