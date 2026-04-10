"""
LSTM model for pharmacy demand forecasting — v2.

Supports standard Sequential LSTM and Bidirectional LSTM.
Includes ModelCheckpoint to save the best epoch automatically.
"""
import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT_RATE, LEARNING_RATE,
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, MODEL_DIR, SEQUENCE_LENGTH,
    USE_BIDIRECTIONAL,
)


def build_model(input_shape: tuple, bidirectional: bool = None) -> Sequential:
    """
    Build the LSTM architecture.

    Args:
        input_shape  : (sequence_length, num_features)
        bidirectional: override config.USE_BIDIRECTIONAL if specified
    """
    if bidirectional is None:
        bidirectional = USE_BIDIRECTIONAL

    model = Sequential()
    model.add(Input(shape=input_shape))

    if bidirectional:
        model.add(Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True)))
    else:
        model.add(LSTM(LSTM_UNITS_1, return_sequences=True))

    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())

    if bidirectional:
        model.add(Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=False)))
    else:
        model.add(LSTM(LSTM_UNITS_2, return_sequences=False))

    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

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
    bidirectional: bool = None,
) -> tuple:
    """
    Train LSTM model for a specific product and save best weights.

    Returns:
        (model, history)
    """
    if epochs     is None: epochs     = EPOCHS
    if batch_size is None: batch_size = BATCH_SIZE

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"lstm_{product_id}.keras")

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, bidirectional=bidirectional)

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
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
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

    print(f"💾 Best model saved → {model_path}")
    return model, history


def load_trained_model(product_id: str):
    """Load a previously trained model for a product."""
    model_path = os.path.join(MODEL_DIR, f"lstm_{product_id}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for {product_id} at {model_path}\n"
            "Run: python src/train.py"
        )
    return keras_load_model(model_path)


def predict_demand(
    model,
    last_sequence: np.ndarray,
    scaler,
    horizon: int = 30,
) -> np.ndarray:
    """
    Multi-step demand forecast using autoregressive prediction.

    Args:
        model        : trained LSTM model
        last_sequence: (sequence_length, num_features) — the last window
        scaler       : fitted MinMaxScaler to inverse-transform
        horizon      : days to forecast

    Returns:
        Array of predicted quantities (non-negative integers)
    """
    predictions  = []
    current_seq  = last_sequence.copy()

    for _ in range(horizon):
        pred     = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)
        pred_val = pred[0, 0]
        predictions.append(pred_val)

        # Slide window: drop oldest, append new row
        new_row    = current_seq[-1].copy()
        new_row[0] = pred_val   # quantity_sold is column 0
        current_seq = np.vstack([current_seq[1:], new_row])

    # Inverse-transform (quantity_sold is column 0)
    predictions = np.array(predictions).reshape(-1, 1)
    n_features  = scaler.n_features_in_
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, 0] = predictions[:, 0]
    inv    = scaler.inverse_transform(dummy)
    demand = inv[:, 0]

    return np.maximum(demand, 0).astype(int)
