import os
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor
from src.models.baselines import BaselineModels

def main():
    csv_path = 'data/Global_Pollution_Analysis.csv'
    
    # Create a mock CSV if it doesn't exist for validation purposes
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Creating a dummy dataset for validation.")
        os.makedirs('data', exist_ok=True)
        np.random.seed(42)
        n_samples = 300
        dummy_data = {
            'Country': np.random.choice(['USA', 'China', 'India', 'Brazil', 'Germany'], n_samples),
            'Year': np.random.choice([2020, 2021, 2022], n_samples),
            'Air_Pollution_Index': np.random.uniform(20, 150, n_samples),
            'Industrial_Waste': np.random.uniform(500, 5000, n_samples),
            'CO2_Emissions': np.random.uniform(1000, 10000, n_samples),
            'GDP': np.random.uniform(5000, 50000, n_samples),
            'Energy_Recovered_GWh': np.random.uniform(10, 500, n_samples)
        }
        df = pd.DataFrame(dummy_data)
        df.loc[0:15, 'Air_Pollution_Index'] = np.nan
        df.to_csv(csv_path, index=False)

    # 1. Initialization
    preprocessor = DataPreprocessor()
    models = BaselineModels()

    # 2. Data Loading and Preprocessing
    df = preprocessor.load_data(csv_path)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, strategy='mean')
    
    # Generate Heatmap
    models.plot_correlation_heatmap(df, save_path='outputs/correlation_heatmap.png')
    print("Heatmap saved to outputs/correlation_heatmap.png")

    # Create Pollution_Severity classification target
    if 'Pollution_Severity' not in df.columns:
        if 'Air_Pollution_Index' in df.columns:
            df['Pollution_Severity'] = pd.qcut(df['Air_Pollution_Index'], q=3, labels=['Low', 'Medium', 'High'])
        elif 'Pollution_Index' in df.columns:
            df['Pollution_Severity'] = pd.qcut(df['Pollution_Index'], q=3, labels=['Low', 'Medium', 'High'])
        else:
            df['Pollution_Severity'] = np.random.choice(['Low', 'Medium', 'High'], len(df))

    # Encode categorical features
    cat_cols = ['Country', 'Year', 'Pollution_Severity']
    df_encoded = preprocessor.encode_features(df, columns=[c for c in cat_cols if c in df.columns])
    
    # Scale numerical features (excluding targets)
    num_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [c for c in num_cols if c not in ['Energy_Recovered_GWh', 'Pollution_Severity']]
    df_scaled = preprocessor.scale_features(df_encoded, columns=cols_to_scale)

    # -- Linear Regression --
    print("\n--- Linear Regression (Target: Energy_Recovered_GWh) ---")
    if 'Energy_Recovered_GWh' in df_scaled.columns:
        # Avoid including classification target
        reg_df = df_scaled.drop(columns=['Pollution_Severity']) if 'Pollution_Severity' in df_scaled.columns else df_scaled
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocessor.split_data(reg_df, target_column='Energy_Recovered_GWh', test_size=0.2)
        
        models.train_linear_regression(X_train_reg, y_train_reg)
        reg_metrics = models.evaluate_linear_regression(X_test_reg, y_test_reg)
        print(f"R2: {reg_metrics['R2']:.4f}")
        print(f"MSE: {reg_metrics['MSE']:.4f}")
        print(f"MAE: {reg_metrics['MAE']:.4f}")

    # -- Logistic Regression --
    print("\n--- Logistic Regression (Target: Pollution_Severity) ---")
    if 'Pollution_Severity' in df_scaled.columns:
        # Avoid including regression target
        class_df = df_scaled.drop(columns=['Energy_Recovered_GWh']) if 'Energy_Recovered_GWh' in df_scaled.columns else df_scaled
        
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = preprocessor.split_data(class_df, target_column='Pollution_Severity', test_size=0.2)
        
        models.train_logistic_regression(X_train_cls, y_train_cls)
        cls_metrics = models.evaluate_logistic_regression(X_test_cls, y_test_cls)
        print(f"Accuracy: {cls_metrics['Accuracy']:.4f}")
        print("Classification Report:")
        print(cls_metrics['Classification_Report'])

if __name__ == "__main__":
    main()
