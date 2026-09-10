import pandas as pd


def load_data(path="data/daily-total-female-births.csv"):

    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()

    print(f"Data Loaded: {df.shape[0]}Row {df.shape[1]} Col")
    print(f"From {df.index.min()} to {df.index.max()}")
    return df


def train_test_splite_temporal(df, train_ratio=0.8):
    """
    Time series split while keeping the time order (no shuffling!)

    """
    train_size = int(len(df) * train_ratio)
    train = df.iloc[: train_size]
    test = df.iloc[train_size:]
    
    print(f"Train: {len(train)},| Test: {len(test)}")
    return train, test

