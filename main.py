import numpy as np
from src.data_loader import load_data, train_test_splite_temporal
from src.feature_engineering import build_all_features
from src import sarima_model
from src import lstm_model
from src.evaluation import calculate_metrics, plot_forecast


def main():
    # 1. Loading Data
    df = load_data('data/daily-total-female-births.csv')

    # 2. Split before Feature Engineering
    train, test = train_test_splite_temporal(df, train_ratio=0.8)

    # 3. Features (just for reporting/analysis - LSTM and SARIMA use the raw series)
    df_features = build_all_features(df)

    # 4. SARIMA
    print("\n" + "="*60)
    print("SARIMA")
    print("="*60)
    params = sarima_model.find_best_params(train['Births'])
    sarima = sarima_model.train(
        train['Births'], params['order'], params['seasonal_order'])
    sarima_model.save(sarima)

    sarima_preds = sarima_model.forecast(sarima, steps=len(test))
    sarima_metrics = calculate_metrics(
        test['Births'].values, sarima_preds.values)
    print(f"SARIMA metrics: {sarima_metrics}")

    # 5. LSTM
    print("\n" + "="*60)
    print("LSTM")
    print("="*60)
    WINDOW = 7
    X_train, y_train, scaler = lstm_model.prepare_data(
        train['Births'].values, window=WINDOW
    )
    lstm, history = lstm_model.train(
        X_train, y_train, window=WINDOW, epochs=30)
    lstm_model.save(lstm, scaler)

    # Prediction on the test using recursive
    last_train_values = train['Births'].values[-WINDOW:]
    lstm_preds = lstm_model.forecast_recursive(
        lstm, scaler, last_train_values, steps=len(test), window=WINDOW
    )
    lstm_metrics = calculate_metrics(
        test['Births'].values, np.array(lstm_preds))
    print(f"LSTM metrics: {lstm_metrics}")

    # 6. Report
    print("\n" + "="*60)
    print("Final comparison")
    print("="*60)
    print(f"SARIMA: {sarima_metrics}")
    print(f"LSTM:   {lstm_metrics}")

    plot_forecast(
        test['Births'],
        {'SARIMA': sarima_preds.values, 'LSTM': lstm_preds},
        save_path='reports/forecast_comparison.png'
    )


if __name__ == '__main__':
    main()
