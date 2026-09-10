import pandas as pd
import numpy as np


def add_time_features(df, date_column=None):
    """
        Time-based features: day of the week, month, quarter
    """

    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df


def add_lag_features(df, target_col, lags=[1, 7, 14, 30]):
    df = df.copy()

    for lag in lags:
        df[f'[lag_{lag}'] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col, window=[7, 14, 30]):
    df = df.copy()

    for w in window:
        df[f'rolling_mean_{w}'] = df[target_col].rolling(w).mean()
        df[f'rolling_std_{w}'] = df[target_col].rolling(w).std()
        df[f'rolling_max_{w}'] = df[target_col].rolling(w).max()
        df[f'rolling_min_{w}'] = df[target_col].rolling(w).min()
    return df


def add_diff_features(df, target_col, periods=[1, 7]):
    df = df.copy()

    for p in periods:
        df[f'diff_{p}'] = df[target_col].diff(p)
    return df


def build_all_features(df, target_col="Births"):
    df = add_time_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_diff_features(df, target_col)
    df = df.dropna()  
    print(f"Features were created. Final form: {df.shape}")
    return df