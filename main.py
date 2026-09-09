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
