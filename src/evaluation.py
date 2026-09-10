import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'MAPE': round(mape, 2)}


def plot_forecast(y_test, predictions_dict, save_path='reports/forecast.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=2)
    
    for name, preds in predictions_dict.items():
        if len(preds) == len(y_test):
            plt.plot(y_test.index, preds, label=name, linestyle='--')
    
    plt.title('Time Series Forecast Comparison', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Chart saved: {save_path}")