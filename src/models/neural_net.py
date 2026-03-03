"""Feedforward Neural Network for Energy Recovery prediction.

Builds, trains, and evaluates a Keras Sequential regression model with
Dropout regularization, learning-rate scheduling, and early stopping.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EnergyRecoveryNN:
    """Feedforward ANN regressor for predicting Energy_Recovered_GWh.

    Attributes:
        model (keras.Sequential): The compiled Keras model.
        history (keras.callbacks.History): Training history after fitting.
        input_dim (int): Number of input features.
    """

    def __init__(self, input_dim: int,
                 hidden_units: list = None,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 1e-3):
        """Initializes the model architecture.

        Args:
            input_dim: Number of input features.
            hidden_units: List of neuron counts per hidden layer.
                          Defaults to [128, 64, 32].
            dropout_rate: Dropout fraction applied after each hidden layer.
            learning_rate: Initial learning rate for Adam optimizer.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build()
        self.history = None

    def _build(self) -> keras.Sequential:
        """Constructs the Sequential model.

        Returns:
            Compiled Keras Sequential model.
        """
        model = keras.Sequential(name='EnergyRecoveryANN')
        model.add(layers.Input(shape=(self.input_dim,), name='input'))

        for i, units in enumerate(self.hidden_units):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))

        model.add(layers.Dense(1, activation='linear', name='output'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 200, batch_size: int = 32) -> keras.callbacks.History:
        """Trains the model with early stopping and LR scheduling.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.

        Returns:
            Keras History object from training.
        """
        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_loss', patience=20,
                restore_best_weights=True, verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=10, min_lr=1e-6, verbose=0
            ),
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb_list,
            verbose=0
        )
        print(f"Training stopped at epoch {len(self.history.history['loss'])}")
        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Computes regression metrics on the test set.

        Args:
            X_test: Test features.
            y_test: Test targets.

        Returns:
            Dict with R2, MSE, MAE keys.
        """
        preds = self.model.predict(X_test, verbose=0).flatten()
        r2  = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        return {'R2': r2, 'MSE': mse, 'MAE': mae}

    def plot_loss_curves(self, save_path: str = 'outputs/loss_curves.png'):
        """Saves training vs. validation loss curves.

        Args:
            save_path: Destination PNG path.
        """
        if self.history is None:
            print("Model not trained yet.")
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        hist = self.history.history

        epochs = range(1, len(hist['loss']) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, hist['loss'], label='Train Loss (MSE)', color='#4C72B0')
        plt.plot(epochs, hist['val_loss'], label='Val Loss (MSE)', color='#DD8452', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training vs. Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss curves saved to {save_path}")

    def summary(self):
        """Prints the model architecture summary."""
        self.model.summary()
