import os
import joblib
import warnings
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings('ignore')


def find_best_params(y_train, seasonal_period=7, max_=5, max_q=5):
    print("Searching for the best SARIMA parameters...")

    auto_model = pm.auto_arima(
        y_train,
        seasonal=True,
        m=seasonal_period,
        start_p=0, max_p=max_p,
        start_q=0, max_q=max_q,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        d=None, D=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        information_criterion='aic'
    )
    
    print(f"\norder: {auto_model.order}")
    print(f"seasonal_order: {auto_model.seasonal_order}")
    print(f"AIC: {auto_model.aic():.2f}")
    
    return {
        'order': auto_model.order,
        'seasonal_order': auto_model.seasonal_order,
        'aic': auto_model.aic()
    }


def train(y_train, order, seasonal_order):
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


def save(model, path='models/sarima_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"The SARIMA model has been saved: {path}")


def load(path='models/sarima_model.pkl'):
    model = joblib.load(path)
    print(f"SARIMA model loaded:{path}")
    return model


def forecast(model, steps):
    return model.forecast(steps=steps)