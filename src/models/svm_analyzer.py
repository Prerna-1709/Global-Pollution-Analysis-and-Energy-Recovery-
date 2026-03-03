"""SVM Analyzer module for deforestation prediction.

Provides SVMAnalyzer with GridSearchCV tuning, feature importance,
evaluation metrics, and scatter plot visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


class SVMAnalyzer:
    """Trains and evaluates an SVM (SVR) for continuous deforestation prediction.

    Attributes:
        best_model: The best estimator found by GridSearchCV.
        grid_search: The fitted GridSearchCV object.
        feature_names (list): Names of input features after fitting.
        feature_importances_ (pd.Series): Feature importances ranked by permutation.
    """

    PARAM_GRID = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
    }

    def __init__(self, cv: int = 5):
        """Initializes SVMAnalyzer.

        Args:
            cv: Number of cross-validation folds. Defaults to 5.
        """
        self.cv = cv
        self.best_model = None
        self.grid_search = None
        self.feature_names = []
        self.feature_importances_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None):
        """Runs GridSearchCV over kernel/C/gamma and fits the best estimator.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.
            feature_names: Column names corresponding to X_train columns.
        """
        self.feature_names = feature_names if feature_names is not None else [f'f{i}' for i in range(X_train.shape[1])]

        base_svr = SVR()
        self.grid_search = GridSearchCV(
            base_svr,
            self.PARAM_GRID,
            cv=self.cv,
            scoring='r2',
            n_jobs=-1,
            verbose=0,
        )
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        print(f"Best Parameters: {self.grid_search.best_params_}")
        print(f"Best CV R2 score: {self.grid_search.best_score_:.4f}")

    def compute_feature_importance(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.Series:
        """Computes permutation-based feature importance on test data.

        Args:
            X_test: Test feature matrix.
            y_test: Test target vector.

        Returns:
            pd.Series of feature importances sorted descending.
        """
        result = permutation_importance(
            self.best_model, X_test, y_test,
            n_repeats=10, random_state=42, scoring='r2'
        )
        self.feature_importances_ = pd.Series(
            result.importances_mean, index=self.feature_names
        ).sort_values(ascending=False)
        return self.feature_importances_

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluates the best model on test data.

        Args:
            X_test: Test feature matrix.
            y_test: Test target vector.

        Returns:
            Dict with keys MAE, MSE, RMSE, R2.
        """
        preds = self.best_model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Runs 5-fold cross-validation on the best model.

        Args:
            X: Full feature matrix.
            y: Full target vector.

        Returns:
            Array of R2 scores for each fold.
        """
        scores = cross_val_score(self.best_model, X, y, cv=self.cv, scoring='r2')
        print(f"Cross-Validation R2 scores ({self.cv} folds): {scores}")
        print(f"Mean CV R2: {scores.mean():.4f}  Std: {scores.std():.4f}")
        return scores

    def plot_feature_importance(self, save_path: str = 'outputs/feature_importance.png'):
        """Saves a bar chart of permutation feature importances.

        Args:
            save_path: Destination file path for the PNG.
        """
        if self.feature_importances_ is None:
            print("Feature importances not computed yet. Call compute_feature_importance() first.")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        colors = ['#4C72B0' if v >= 0 else '#DD8452' for v in self.feature_importances_.values]
        self.feature_importances_.plot(kind='barh', color=colors)
        plt.xlabel('Permutation Importance (R2 decrease)')
        plt.title('Feature Importance for Deforestation Prediction (SVM)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Feature importance chart saved to {save_path}")

    def plot_top_feature_scatter(self, X_test: np.ndarray, y_test: np.ndarray,
                                  save_path: str = 'outputs/top_feature_scatter.png'):
        """Scatter plot of the top feature vs actual and predicted target values.

        Args:
            X_test: Test feature matrix (numpy array).
            y_test: Test target labels.
            save_path: Destination file path for the PNG.
        """
        if self.feature_importances_ is None:
            print("Feature importances not computed. Call compute_feature_importance() first.")
            return

        top_feat = self.feature_importances_.index[0]
        feat_idx = list(self.feature_names).index(top_feat)
        x_vals = X_test[:, feat_idx]
        preds = self.best_model.predict(X_test)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(9, 6))
        plt.scatter(x_vals, y_test, alpha=0.6, label='Actual', color='#4C72B0')
        plt.scatter(x_vals, preds, alpha=0.6, label='Predicted', color='#DD8452', marker='x')
        plt.xlabel(top_feat)
        plt.ylabel('Forest Loss Target')
        plt.title(f'Top Feature ({top_feat}) vs Forest Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Scatter plot saved to {save_path}")
