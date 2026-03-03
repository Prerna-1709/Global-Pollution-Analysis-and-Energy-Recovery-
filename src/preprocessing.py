"""Data preprocessing module for Global Pollution Analysis."""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Class to handle missing values, categorical encoding, and feature scaling."""

    def __init__(self):
        """Initializes the DataPreprocessor with necessary transformers."""
        self.imputers = {}
        self.scalers = {}
        self.label_encoders = {}
        self.onehot_encoder = None

    def load_data(self, file_path):
        """
        Loads CSV files into a pandas DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        return pd.read_csv(file_path)

    def handle_missing_values(self, df, strategy='mean'):
        """
        Identifies and imputes missing values for numerical features and handles categorical anomalies.

        Args:
            df (pd.DataFrame): The input DataFrame.
            strategy (str, optional): The imputation strategy for numerical features. Defaults to 'mean'.

        Returns:
            pd.DataFrame: The DataFrame with missing values handled.
        """
        df_processed = df.copy()
        
        # Handle numerical columns
        num_cols = df_processed.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            num_imputer = SimpleImputer(strategy=strategy)
            df_processed[num_cols] = num_imputer.fit_transform(df_processed[num_cols])
            self.imputers['numerical'] = num_imputer

        # Handle categorical columns
        cat_cols = df_processed.select_dtypes(exclude=[np.number]).columns
        if not cat_cols.empty:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[cat_cols] = cat_imputer.fit_transform(df_processed[cat_cols])
            self.imputers['categorical'] = cat_imputer

        return df_processed

    def encode_features(self, df, columns):
        """
        Applies LabelEncoder or OneHotEncoder to categorical features.
        Uses LabelEncoder as a straightforward approach for specified columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (list): List of column names to encode.

        Returns:
            pd.DataFrame: The DataFrame with encoded categorical features.
        """
        df_encoded = df.copy()
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        return df_encoded

    def scale_features(self, df, columns):
        """
        Applies StandardScaler to numerical features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (list): List of columns to scale.

        Returns:
            pd.DataFrame: The DataFrame with scaled numerical features.
        """
        df_scaled = df.copy()
        
        cols_to_scale = [col for col in columns if col in df_scaled.columns]
        if cols_to_scale:
            scaler = StandardScaler()
            df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
            self.scalers['standard'] = scaler
            
        return df_scaled

    def split_data(self, df, target_column, test_size=0.2):
        """
        Splits data into training and testing sets.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target variable column.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
        else:
            raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")


if __name__ == "__main__":
    print("Testing DataPreprocessor with dummy dataframe...")
    
    # Create dummy dataframe
    dummy_data = {
        'Country': ['USA', 'China', 'India', 'USA', np.nan, 'China'],
        'Year': ['2020', '2020', '2021', np.nan, '2021', '2022'],
        'CO2_Emissions': [5000.5, 9000.1, np.nan, 4500.0, 3000.2, 9500.9],
        'GDP': [21000, 14000, 2800, 22000, np.nan, 15000],
        'Pollution_Index': [45.2, 85.1, 75.0, 42.1, 60.5, np.nan],
        'Target': [1, 0, 1, 1, 0, 0]
    }
    
    df = pd.DataFrame(dummy_data)
    print("Original DataFrame:\n", df)
    print("-" * 50)
    
    preprocessor = DataPreprocessor()
    
    # 1. Handle missing values
    df_clean = preprocessor.handle_missing_values(df, strategy='mean')
    print("Missing values handled. Null counts:\n", df_clean.isnull().sum())
    
    # 2. Encode categorical features
    df_encoded = preprocessor.encode_features(df_clean, columns=['Country', 'Year'])
    print("DataFrame after encoding:\n", df_encoded[['Country', 'Year']].head())
    
    # 3. Scale features
    df_scaled = preprocessor.scale_features(df_encoded, columns=['CO2_Emissions', 'GDP', 'Pollution_Index'])
    print("DataFrame after scaling:\n", df_scaled[['CO2_Emissions', 'GDP', 'Pollution_Index']].head())
    
    # 4. Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_scaled, target_column='Target', test_size=0.2)
    print("-" * 50)
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print("Confirmation: No null values in X_train ->", X_train.isnull().sum().sum() == 0)
