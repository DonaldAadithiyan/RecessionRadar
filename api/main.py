import pandas as pd
import numpy as np
from pydantic import BaseModel
from ML_pipe import time_series_prediction, regresstion_feature_engineering, regression_prediction

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
    base_pred: float
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
    "base_pred": None,
    "one_month": None,
    "three_month": None,
    "six_month": None,
    "updated_at": None
}

yields = {}
indicators = {}
recession_data = None

def run_ml_pipeline_periodically():
    global latest_predictions, yields, indicators, recession_data
    while True:
        ## ML PIPELINE
        # time_series_prediction()
        fe_data = regresstion_feature_engineering()
        try:
            base, one_month, three_month, six_month= regression_prediction(fe_data)
            latest_predictions = {
                "base_pred": base,
                "one_month": one_month,
                "three_month": three_month,
                "six_month": six_month,
                "updated_at": datetime.now().isoformat()
            }
            print("ML pipeline updated predictions:", latest_predictions)
        except Exception as e:
            print("Error in ML pipeline:", e)
        
        ## Fetch Treasury Yields and Economic Indicators
        if not FRED_API_KEY:
            raise HTTPException(status_code=500, detail="FRED API key not set in environment variable FRED_API_KEY")

        def fetch(label_series):
            label, series_id = label_series
            value = fetch_latest_fred_value(series_id)
            value = float(value) if value is not None else None
            return (label, value)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch, FRED_SERIES.items()))
            for label, value in results:
                yields[label] = value
            
            results = list(executor.map(fetch, ECON_FRED_SERIES.items()))
            for label, value in results:
                indicators[label] = value
        print("Updated yields:", yields)
        print("Updated indicators:", indicators)
        
        ## recesstion dataset
        if os.path.exists('../data/fix'):
            train_df = pd.read_csv('../data/fix/feature_selected_recession_train.csv')
            test_df = pd.read_csv('../data/fix/feature_selected_recession_test.csv')
        else:
            train_df = pd.read_csv('data/fix/feature_selected_recession_train.csv')
            test_df = pd.read_csv('data/fix/feature_selected_recession_test.csv')

        # Ensure 'date' is datetime
        train_df['date'] = pd.to_datetime(train_df['date'])
        test_df['date'] = pd.to_datetime(test_df['date'])

        # Combine into one DataFrame
        full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        full_df = full_df.sort_values('date').reset_index(drop=True)
        
        # Replace NaN, inf, -inf with 0.0
        full_df = full_df.replace([np.nan, np.inf, -np.inf], 0.0)
        
        recession_data = full_df[["date", "recession_probability", "1_month_recession_probability", 
                                  "3_month_recession_probability", "6_month_recession_probability"]].copy() 
        print("Updated recession data:")  
        
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
    return TreasuryYields(
        yields=yields,
        updated_at=datetime.now().isoformat()
    )

# Get Economic Indicators
@app.get("/api/economic-indicators", response_model=EconomicIndicators)
async def get_economic_indicators():    
    return EconomicIndicators(
        indicators=indicators,
        updated_at=datetime.now().isoformat()
    )

# Get Recession Probabilities from FRED API and generate shifted columns
@app.get("/api/recession-probabilities", response_model=RecessionProbabilities)
async def get_recession_probabilities():
    return RecessionProbabilities(
        dates = [d.strftime("%Y-%m-%d") for d in recession_data["date"]],
        base = recession_data["recession_probability"].tolist(),
        one_month = recession_data["1_month_recession_probability"].tolist(),
        three_month = recession_data["3_month_recession_probability"].tolist(),
        six_month = recession_data["6_month_recession_probability"].tolist()
    )


# Get Current Recession Prediction
@app.get("/api/current-prediction", response_model=RecessionPrediction)
async def get_current_prediction():
    return RecessionPrediction(**latest_predictions)


# Custom Prediction endpoint
@app.post("/api/custom-prediction", response_model=RecessionPrediction)
async def create_custom_prediction(request: CustomPredictionRequest):
    inputs = request.indicators
    
    custom_data = regresstion_feature_engineering(inputs)
    base_pred, one_month, three_month, six_month = regression_prediction(custom_data)
    print(inputs)
    print(f"Custom Prediction - Base: {base_pred}, 1m: {one_month}, 3m: {three_month}, 6m: {six_month}")
    return RecessionPrediction(
        base_pred = min(max(base_pred, 0), 1),
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
