import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

class BaselineModels:
    """Class to encapsulate baseline prediction models: Linear and Logistic Regression."""
    
    def __init__(self):
        self.linear_reg = LinearRegression()
        self.logistic_reg = LogisticRegression(max_iter=1000)

    def train_linear_regression(self, X_train, y_train):
        """Trains the Linear Regression model."""
        self.linear_reg.fit(X_train, y_train)

    def evaluate_linear_regression(self, X_test, y_test):
        """Evaluates the Linear Regression model and returns metrics."""
        predictions = self.linear_reg.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        return {'R2': r2, 'MSE': mse, 'MAE': mae}

    def train_logistic_regression(self, X_train, y_train):
        """Trains the Logistic Regression model."""
        self.logistic_reg.fit(X_train, y_train)

    def evaluate_logistic_regression(self, X_test, y_test):
        """Evaluates the Logistic Regression model and returns classification metrics."""
        predictions = self.logistic_reg.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average='macro', zero_division=0)
        rec = recall_score(y_test, predictions, average='macro', zero_division=0)
        f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
        report = classification_report(y_test, predictions, zero_division=0)
        return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'Classification_Report': report}

    def plot_correlation_heatmap(self, df, save_path='outputs/correlation_heatmap.png'):
        """Plots and saves a correlation heatmap for numerical features."""
        plt.figure(figsize=(10, 8))
        numeric_cols = df.select_dtypes(include=[np.number])
        corr = numeric_cols.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
