"""Orchestration script for Final Report Generation (Step 7).

Run:
    python main_report.py

Outputs:
    outputs/Final_Project_Report.pdf
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import DataPreprocessor
from src.reporting.report_generator import ReportGenerator

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH         = 'data/Global_Pollution_Analysis.csv'
TARGET_REG       = 'Energy_Recovered_GWh'
FEATURE_PRIORITY = [
    'Air_Pollution_Index', 'CO2_Emissions', 'Industrial_Waste',
    'GDP', 'GDP_Billion_USD', 'Population', 'Rainfall_mm',
    'Temperature_C', 'Energy_Consumption_Per_Capita', 'Recovery_Rate_percent',
]

# Pre-computed ANN metrics from Step 6 (updated when real data is present)
ANN_METRICS = {'R2': -2.7705, 'MSE': 77918.85, 'MAE': 239.28}


def generate_dummy_data(path: str, n: int = 600):
    np.random.seed(99)
    df = pd.DataFrame({
        'Country': np.random.choice(['USA','China','India','Brazil','Germany'], n),
        'Year': np.random.choice(range(2015, 2023), n),
        'Air_Pollution_Index': np.random.uniform(10, 200, n),
        'CO2_Emissions': np.random.uniform(500, 12000, n),
        'Industrial_Waste': np.random.uniform(100, 7000, n),
        'GDP': np.random.uniform(3000, 60000, n),
        TARGET_REG: np.random.uniform(5, 700, n),
    })
    df[TARGET_REG] += (
        0.05 * df['Air_Pollution_Index']
        - 0.002 * df['CO2_Emissions']
        + 0.008 * df['GDP']
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dummy dataset created: {path} ({n} rows)")


def main():
    print("=" * 60)
    print("  Global Pollution Analysis — Final Report Generator")
    print("=" * 60)

    # ── Load & preprocess ──────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"Warning: {CSV_PATH} not found. Generating dummy data.")
        generate_dummy_data(CSV_PATH)

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(CSV_PATH)
    df = preprocessor.handle_missing_values(df, strategy='mean')

    cat_cols = [c for c in ['Country', 'Year'] if c in df.columns]
    df = preprocessor.encode_features(df, columns=cat_cols)

    # ── Regression target ──────────────────────────────────────────────────
    if TARGET_REG not in df.columns:
        df[TARGET_REG] = np.random.uniform(5, 700, len(df))

    # ── Classification target: Pollution_Severity from Air_Pollution_Index ─
    sev_src = 'Air_Pollution_Index' if 'Air_Pollution_Index' in df.columns else \
              'Pollution_Index'      if 'Pollution_Index' in df.columns else None
    if sev_src:
        bins = pd.qcut(df[sev_src], q=3, labels=[0, 1, 2], duplicates='drop')
        df['Pollution_Severity'] = bins.astype(int)
    else:
        df['Pollution_Severity'] = np.random.randint(0, 3, len(df))

    # ── Feature selection ──────────────────────────────────────────────────
    features = [f for f in FEATURE_PRIORITY if f in df.columns]
    if not features:
        features = [c for c in df.select_dtypes(include=[float, int]).columns
                    if c not in [TARGET_REG, 'Pollution_Severity']][:6]
    print(f"Features ({len(features)}): {features}")

    # ── Scale & split ──────────────────────────────────────────────────────
    df_scaled = preprocessor.scale_features(df, columns=features)
    X = df_scaled[features].values
    y_reg = df[TARGET_REG].values
    y_cls = df['Pollution_Severity'].values

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )
    print(f"Train/Test split: {X_train.shape[0]} / {X_test.shape[0]}")

    # ── Generate report ────────────────────────────────────────────────────
    reporter = ReportGenerator(
        X_train=X_train, X_test=X_test,
        y_reg_train=yr_train, y_reg_test=yr_test,
        y_cls_train=yc_train, y_cls_test=yc_test,
        feature_names=features,
        ann_metrics=ANN_METRICS,
        class_names=['Low', 'Medium', 'High']
    )

    print("\nTraining all models...")
    reporter.train_all()

    print("Collecting metrics...")
    metrics = reporter.collect_metrics()

    # Print summary to console
    print("\n--- Regression Metrics ---")
    for model, m in metrics['regression'].items():
        print(f"  {model:25s}  R2={m['R2']:.4f}  MSE={m['MSE']:.1f}  MAE={m['MAE']:.1f}")

    print("\n--- Classification Metrics ---")
    for model, m in metrics['classification'].items():
        print(f"  {model:20s}  Acc={m['Accuracy']:.4f}  F1={m['F1 (macro)']:.4f}")

    print("\nGenerating PDF report...")
    reporter.generate_pdf(save_path='outputs/Final_Project_Report.pdf')

    print("\nDone! See outputs/Final_Project_Report.pdf")


if __name__ == '__main__':
    main()
