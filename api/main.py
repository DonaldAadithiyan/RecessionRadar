import pandas as pd
import numpy as np
from pydantic import BaseModel
from reg_FE import feature_eng
from ML_pipe import load_models

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import os, time, random
import json, requests
import dotenv
import threading, concurrent.futures
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


dotenv.load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="RecessionRadar API",
    description="API for recession probability predictions and economic data",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response data
class TreasuryYields(BaseModel):
    yields: Dict[str, float]
    updated_at: str

class EconomicIndicators(BaseModel):
    indicators: Dict[str, float]
    updated_at: str

class RecessionProbabilities(BaseModel):
    dates: List[str]
    one_month: List[float]
    three_month: List[float]
    six_month: List[float]

class RecessionPrediction(BaseModel):
    one_month: float
    three_month: float
    six_month: float
    updated_at: str

class CustomPredictionRequest(BaseModel):
    indicators: Dict[str, float]
    
    
FRED_API_KEY = os.getenv("FRED_API_KEY")

FRED_SERIES = {
    "3-Month Rate": "DTB3",
    "6-Month Rate": "DTB6",
    "1-Year Rate": "DTB1YR",
    "2-Year Rate": "DGS2",
    "5-Year Rate": "DGS5",
    "10-Year Rate": "DGS10",
    "30-Year Rate": "DGS30"
}

ECON_FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "PPI": "PPIACO",
    "Industrial Production": "INDPRO",
    "Share Price": "SP500",
    "Unemployment Rate": "UNRATE",
    "OECD CLI Index": "OECDLOLITOAASTSAM",
    "CSI Index": "UMCSENT"
}


# Global variable to store latest predictions
latest_predictions = {
    "one_month": None,
    "three_month": None,
    "six_month": None,
    "updated_at": None
}

def run_ml_pipeline_periodically():
    global latest_predictions
    one_month_model, three_month_model, six_month_model = load_models()
    while True:
        try:
            # Prepare your input data for the pipeline here
            # For example, fetch latest indicators from FRED or your DB
            # inputs = ...
            # dataset = feature_eng(inputs)
            # one_month_model, three_month_model, six_month_model = load_models()
            # one_month = one_month_model.predict(dataset)[0]
            # three_month = three_month_model.predict(dataset)[0]
            # six_month = six_month_model.predict(dataset)[0]
            # For demo, use random values:
            one_month = random.uniform(0, 1)
            three_month = random.uniform(0, 1)
            six_month = random.uniform(0, 1)
            latest_predictions = {
                "one_month": one_month,
                "three_month": three_month,
                "six_month": six_month,
                "updated_at": datetime.now().isoformat()
            }
            print("ML pipeline updated predictions:", latest_predictions)
        except Exception as e:
            print("Error in ML pipeline:", e)
        time.sleep(300)  # 5 minutes

def fetch_latest_fred_value(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    if "observations" in data and len(data["observations"]) > 0:
        value = data["observations"][0]["value"]
        try:
            return float(value)
        except ValueError:
            return None
    return None

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to RecessionRadar API"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=run_ml_pipeline_periodically, daemon=True)
    thread.start()
    yield

app.router.lifespan_context = lifespan

# Get Treasury Yields
@app.get("/api/treasury-yields", response_model=TreasuryYields)
async def get_treasury_yields():
    if not FRED_API_KEY:
        raise HTTPException(status_code=500, detail="FRED API key not set in environment variable FRED_API_KEY")

    yields = {}
    
    def fetch(label_series):
        label, series_id = label_series
        value = fetch_latest_fred_value(series_id)
        value = float(value) if value is not None else None
        return (label, value)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch, FRED_SERIES.items()))
        for label, value in results:
            yields[label] = value
            
    print(yields)
    return TreasuryYields(
        yields=yields,
        updated_at=datetime.now().isoformat()
    )

# Get Economic Indicators
@app.get("/api/economic-indicators", response_model=EconomicIndicators)
async def get_economic_indicators():
    if not FRED_API_KEY:
        raise HTTPException(status_code=500, detail="FRED API key not set in environment variable FRED_API_KEY")

    indicators = {}

    def fetch(label_series):
        label, series_id = label_series
        value = fetch_latest_fred_value(series_id)
        value = float(value) if value is not None else None
        return (label, value)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch, ECON_FRED_SERIES.items()))
        for label, value in results:
            indicators[label] = value
            
    print(indicators)
    
    return EconomicIndicators(
        indicators=indicators,
        updated_at=datetime.now().isoformat()
    )

# Get Recession Probabilities from FRED API and generate shifted columns
@app.get("/api/recession-probabilities", response_model=RecessionProbabilities)
async def get_recession_probabilities():
    if not FRED_API_KEY:
        raise HTTPException(status_code=500, detail="FRED API key not set in environment variable FRED_API_KEY")
    # Fetch the RECPROUSM156N series from FRED
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "RECPROUSM156N",
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch data from FRED API.")
    data = resp.json()
    if "observations" not in data:
        raise HTTPException(status_code=500, detail="FRED API response missing observations.")
    # Convert to DataFrame
    df = pd.DataFrame(data["observations"])
    if "date" not in df.columns or "value" not in df.columns:
        raise HTTPException(status_code=500, detail="FRED data missing required columns.")
    df = df[["date", "value"]].copy()
    df["recession_probability"] = pd.to_numeric(df["value"], errors="coerce")
    # Create shifted columns
    df["1_month_recession_probability"] = df["recession_probability"].shift(-1)
    df["3_month_recession_probability"] = df["recession_probability"].shift(-3)
    df["6_month_recession_probability"] = df["recession_probability"].shift(-6)
    # Calculate the range (min, max) for each probability column
    # Drop rows with NaN in any probability column
    df = df.dropna(subset=["1_month_recession_probability", "3_month_recession_probability", "6_month_recession_probability"])
    return RecessionProbabilities(
        dates=df["date"].tolist(),
        one_month=df["1_month_recession_probability"].tolist(),
        three_month=df["3_month_recession_probability"].tolist(),
        six_month=df["6_month_recession_probability"].tolist()
    )

# Get Current Recession Prediction
@app.get("/api/current-prediction", response_model=RecessionPrediction)
async def get_current_prediction():
    return RecessionPrediction(**latest_predictions)
    # return RecessionPrediction(
    #     # forcast=1/100,
    #     one_month=0.5/100,
    #     three_month = 12/100,
    #     six_month = 80/100,
    #     updated_at=datetime.now().isoformat()
    # )

# Custom Prediction endpoint
@app.post("/api/custom-prediction", response_model=RecessionPrediction)
async def create_custom_prediction(request: CustomPredictionRequest):
    inputs = request.indicators
    dataset = feature_eng(inputs)
    
    one_month_model, three_month_model, six_month_model = load_models()

    one_month = one_month_model.predict(dataset)[0]
    three_month = three_month_model.predict(dataset)[0]
    six_month = six_month_model.predict(dataset)[0]

    return RecessionPrediction(
        # forcast = min(max(forcast, 0), 1),
        one_month=min(max(one_month, 0), 1),
        three_month=min(max(three_month, 0), 1),
        six_month=min(max(six_month, 0), 1),
        updated_at=datetime.now().isoformat()
    )

# For development testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # print(load_models)
