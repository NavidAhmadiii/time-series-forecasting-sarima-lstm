from fastapi import FasAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timedelta
import sys
import os
from src import sarima_model, lstm_model


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


app = FasAPI(
    title="Time Series Forecasting API",
    description="Time Series Forecasting with SARIMA and LSTM",
    version="0.0.1"
)

sarima = None
lstm = None
lstm_scaler = None


@app.on_event("startup")
async def load_models():
    global sarima, lstm, lstm_scaler

    try:
        sarima = sarima_model.load()
    except Exception as e:
        print(f"SARIMA: {e}")

    try:
        lstm, lstm_scaler = lstm_model.load()
    except Exception as e:
        print(f"LSTM: {e}")


class ForcastRequest(BaseModel):
    steps: int = Field(7, ge=1, le=90)


class LSTMForecastRequest(BaseModel):
    recent_values: List[float] = Field(..., min_length=7, max_length=7)
    steps: int = Field(1, ge=1, le=30)


@app.get("/")
def root():
    return {
        "message": "TS Forecasting API",
        "docs": "/docs",
        "sarima_loaded": sarima is not None,
        "lstm_loaded": lstm is not None
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.post("/forecast/sarima")
def forecast_sarima(req: ForecastRequest):
    if sarima is None:
        raise HTTPException(503, "SARIMA not loaded")
    preds = sarima_model.forecast(sarima, steps=req.steps).tolist()
    today = datetime.now()
    dates = [(today + timedelta(days=i+1)).strftime('%Y-%m-%d')
             for i in range(req.steps)]
    return {"model": "sarima", "predictions": preds, "dates": dates}


@app.post("/forecast/lstm")
def forecast_lstm(req: LSTMForecastRequest):
    if lstm is None:
        raise HTTPException(503, "LSTM not loaded")
    preds = lstm_model.forecast_recursive(
        lstm, lstm_scaler, req.recent_values, steps=req.steps, window=7
    )
    return {"model": "lstm", "predictions": preds}
