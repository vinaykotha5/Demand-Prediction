"""
LSTM model for pharmacy demand forecasting.
"""
import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info logs

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DROPOUT_RATE,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    MODEL_DIR,
    SEQUENCE_LENGTH,
)


def build_model(input_shape: tuple) -> Sequential:
    """Build the LSTM architecture.

    Args:
        input_shape: (sequence_length, num_features)
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(LSTM_UNITS_1, return_sequences=True),
            Dropout(DROPOUT_RATE),
            LSTM(LSTM_UNITS_2, return_sequences=False),
            Dropout(DROPOUT_RATE),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    product_id: str,
    epochs: int = None,
    batch_size: int = None,
) -> tuple:
    """Train LSTM model for a specific product and save weights.

    Returns:
        (model, history)
    """
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"lstm_{product_id}.keras")

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(model_path)
    print(f"💾 Model saved: {model_path}")

    return model, history


def load_trained_model(product_id: str):
    """Load a previously trained model for a product."""
    model_path = os.path.join(MODEL_DIR, f"lstm_{product_id}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for {product_id} at {model_path}")
    return keras_load_model(model_path)


def predict_demand(
    model,
    last_sequence: np.ndarray,
    scaler,
    horizon: int = 30,
) -> np.ndarray:
    """Predict demand for the next `horizon` days.

    Args:
        model: trained LSTM model
        last_sequence: shape (sequence_length, num_features) — the last window
        scaler: fitted MinMaxScaler to inverse-transform predictions
        horizon: number of days to forecast

    Returns:
        Array of predicted quantities for each day
    """
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(horizon):
        # Predict next value
        pred = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)
        pred_val = pred[0, 0]
        predictions.append(pred_val)

        # Shift window forward: drop first row, append predicted row
        new_row = current_seq[-1].copy()
        new_row[0] = pred_val  # update quantity_sold (col 0)

        # Increment time features slightly (approximation)
        # In production, you'd compute exact date features
        current_seq = np.vstack([current_seq[1:], new_row])

    # Inverse transform predictions (quantity_sold is column 0)
    predictions = np.array(predictions).reshape(-1, 1)

    # Build dummy array matching scaler shape for inverse transform
    num_features = scaler.n_features_in_
    dummy = np.zeros((len(predictions), num_features))
    dummy[:, 0] = predictions[:, 0]
    inv = scaler.inverse_transform(dummy)
    demand = inv[:, 0]

    return np.maximum(demand, 0).astype(int)
