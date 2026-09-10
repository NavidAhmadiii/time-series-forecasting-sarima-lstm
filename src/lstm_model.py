import os
import joblib
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler


def create_sequence(data, window=7):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build(window=7):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(
            window, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def prepare_data(train_series, window=7):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()

    X, y = create_sequences(scaled_train, window)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler


def train(X_train, y_train, window=7, epochs=30, batch_size=16):
    model = build(window)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1
    )
    return model, history


def save(model, scaler, path_prefix='models/lstm'):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    model.save(f'{path_prefix}_model.h5')
    joblib.dump(scaler, f'{path_prefix}_scaler.pkl')
    print(f" LSTM Model Saved {path_prefix}")


def load(path_prefix='models/lstm'):
    model = load_model(f'{path_prefix}_model.h5')
    scaler = joblib.load(f'{path_prefix}_scaler.pkl')
    print(f" LSTM Loaded")
    return model, scaler


def forecast_recursive(model, scaler, recent_values, steps, window=7):
    values = np.array(recent_values).reshape(-1, 1)
    scaled = scaler.transform(values).flatten()

    predictions = []
    seq = scaled.copy()

    for _ in range(steps):
        X = seq.reshape(1, window, 1)
        pred_scaled = model.predict(X, verbose=0)[0, 0]
        predictions.append(pred_scaled)
        seq = np.append(seq[1:], pred_scaled)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    return predictions.tolist()
