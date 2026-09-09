import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings


warnings.filterwarnings('ignore')


# Data Loading
df = pd.read_csv('daily-total-female-births.csv',
                 parse_dates=['Date'], index_col='Date')
print('Dataset Loaded ...')
print(df.head())


# Feature Engineering
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['log_1'] = df['Births'].shift(1)
df['log_7'] = df['Births'].shift(7)
df['rolling_mean_7'] = df['Births'].rolling(7).mean()
df['rolling_std_7'] = df['Births'].rolling(7).std()
df['diff_1'] = df['Births'].diff(1)


df = df.dropna()
print('\n\nNew Feature Created ...')
print(df.head())


# Data split (chronologically)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

y_train = train['Births']
y_test = test['Births']


# SARIMA Model (with simple parameters)
model_sarima = ARIMA(y_train, order=(5, 1, 0))  # Simple (p,d,q)
model_fit = model_sarima.fit()
pred_sarima = model_fit.forecast(steps=len(y_test))


# LSTM Model
def create_sequence(data, window=7):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


scalere = StandardScaler()
scalere_data = scalere.fit_transform(df[['Births']])

X, y = create_sequence(scalere_data.flatten(), window=7)

# train and test Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train_lstm, y_test_lstm = y[:split], y[split:]


# Reshape LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Create LSTM Model
model_lstm = Sequential([
    LSTM(
        50,
        activation='relu',
        input_shape=(7, 1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train_lstm, epochs=20, verbose=0)

# Prediction
pred_lstm_scaled = model_lstm.predict(X_test)
pred_lstm = scalere.inverse_transform(pred_lstm_scaled).flatten()


# Evaluation
actual_lstm = scalere.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

rmse_sarima = np.sqrt(mean_squared_error(y_test, pred_sarima))
mae_sarima = mean_absolute_error(y_test, pred_sarima)


rmse_lstm = np.sqrt(mean_squared_error(actual_lstm, pred_lstm))
mae_lstm = mean_absolute_error(actual_lstm, pred_lstm)


print('Results of  Evaluation:\n\n')
print(f"SARIMA  -> RMSE: {rmse_sarima:.2f}, MAE: {mae_sarima:.2f}")
print(f"LSTM    -> RMSE: {rmse_lstm:.2f}, MAE: {mae_lstm:.2f}")


# Simpler Reporting (Chart)
plt.figure(figsize=(12,5))
plt.plot(y_test.index, y_test, label='Real Value', color='black')
plt.plot(y_test.index, pred_sarima, label='SARIMA Prediction', linestyle='--')

lstm_index = y_test.index[-len(pred_lstm):]
plt.plot(lstm_index, pred_lstm, label='LSTM Prediction', linestyle=':')
plt.title('Daily birth prediction comparison')
plt.xlabel('Date')
plt.ylabel('Number of Births')
plt.legend()
plt.grid(True)
plt.show()

print("\n\nProject Has been Completed!")