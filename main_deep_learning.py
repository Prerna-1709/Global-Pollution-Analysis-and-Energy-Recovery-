"""Orchestration script for Neural Network Energy Recovery Prediction (Step 6).

Run:
    python main_deep_learning.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.preprocessing import DataPreprocessor
from src.models.neural_net import EnergyRecoveryNN

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = 'data/Global_Pollution_Analysis.csv'
TARGET_COL = 'Energy_Recovered_GWh'
FEATURE_CANDIDATES = [
    'Air_Pollution_Index', 'CO2_Emissions', 'Industrial_Waste',
    'GDP', 'Population', 'Rainfall_mm', 'Temperature_C',
    'Energy_Consumption_Per_Capita', 'Recovery_Rate_percent'
]


def generate_dummy_data(path: str, n: int = 600):
    """Creates a synthetic dataset if the real CSV is absent."""
    np.random.seed(42)
    df = pd.DataFrame({
        'Country': np.random.choice(['USA', 'China', 'India', 'Brazil', 'Germany'], n),
        'Year': np.random.choice(range(2015, 2023), n),
        'Air_Pollution_Index': np.random.uniform(10, 200, n),
        'CO2_Emissions': np.random.uniform(500, 12000, n),
        'Industrial_Waste': np.random.uniform(100, 7000, n),
        'GDP': np.random.uniform(3000, 60000, n),
        TARGET_COL: np.random.uniform(5, 700, n),
    })
    # Introduce mild correlations so the NN can learn something
    df[TARGET_COL] += (
        0.03 * df['Air_Pollution_Index']
        - 0.005 * df['CO2_Emissions']
        + 0.01 * df['GDP']
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dummy dataset created at {path} ({n} rows).")


def baseline_linear(X_train, y_train, X_test, y_test) -> dict:
    """Trains a Linear Regression baseline and returns metrics."""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    return {
        'R2':  r2_score(y_test, preds),
        'MSE': mean_squared_error(y_test, preds),
        'MAE': mean_absolute_error(y_test, preds),
    }


def print_comparison(lr_metrics: dict, nn_metrics: dict):
    """Prints a side-by-side comparison table."""
    print("\n" + "=" * 52)
    print(f"{'Metric':<10} {'Linear Regression':>18} {'Neural Network':>18}")
    print("-" * 52)
    for key in ['R2', 'MSE', 'MAE']:
        lr_val = lr_metrics.get(key, float('nan'))
        nn_val = nn_metrics.get(key, float('nan'))
        better = "<-- better" if (
            (key == 'R2' and nn_val > lr_val) or
            (key != 'R2' and nn_val < lr_val)
        ) else ""
        print(f"{key:<10} {lr_val:>18.4f} {nn_val:>18.4f}  {better}")
    print("=" * 52)


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"Warning: {CSV_PATH} not found. Generating dummy dataset.")
        generate_dummy_data(CSV_PATH)

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(CSV_PATH)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    df = preprocessor.handle_missing_values(df, strategy='mean')

    # Encode any categorical columns
    cat_cols = [c for c in ['Country', 'Year'] if c in df.columns]
    df = preprocessor.encode_features(df, columns=cat_cols)

    # Select available features
    features = [f for f in FEATURE_CANDIDATES if f in df.columns]
    if not features:
        features = [c for c in df.select_dtypes(include=[float, int]).columns if c != TARGET_COL][:6]

    if TARGET_COL not in df.columns:
        print(f"Error: target column '{TARGET_COL}' not found. Exiting.")
        return

    print(f"Features ({len(features)}): {features}")

    # Scale
    df_scaled = preprocessor.scale_features(df, columns=features)

    X = df_scaled[features].values
    y = df_scaled[TARGET_COL].values if TARGET_COL in df_scaled.columns else df[TARGET_COL].values

    # ── Splits: 64% train / 16% val / 20% test ───────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=42)
    print(f"Train/Val/Test: {X_train.shape[0]} / {X_val.shape[0]} / {X_test.shape[0]}")

    # ── Baseline: Linear Regression ───────────────────────────────────────────
    print("\n--- Baseline: Linear Regression ---")
    lr_metrics = baseline_linear(X_train, y_train, X_test, y_test)
    for k, v in lr_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Neural Network ────────────────────────────────────────────────────────
    print("\n--- Neural Network (ANN) ---")
    nn = EnergyRecoveryNN(
        input_dim=X_train.shape[1],
        hidden_units=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=1e-3
    )
    nn.summary()

    nn.fit(X_train, y_train, X_val, y_val, epochs=200, batch_size=32)

    nn_metrics = nn.evaluate(X_test, y_test)
    print("\nNeural Network Test Metrics:")
    for k, v in nn_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print_comparison(lr_metrics, nn_metrics)

    # ── Plots ─────────────────────────────────────────────────────────────────
    nn.plot_loss_curves(save_path='outputs/loss_curves.png')

    print("\nDone. All outputs saved to outputs/")


if __name__ == '__main__':
    main()
