from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
import requests
import dotenv
import concurrent.futures


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
# FRED_API_KEY = "b9aad5fff9989e57826ce9229a0f0bdb"

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

# Get Recession Probabilities
@app.get("/api/recession-probabilities", response_model=RecessionProbabilities)
async def get_recession_probabilities(months: int = 24):
    # Generate synthetic data similar to your Streamlit app
    today = datetime.now()
    dates = [(today - timedelta(days=30*i)).strftime("%Y-%m-%d") for i in range(months)]
    dates.reverse()
    
    # Create probabilities with some randomness and trends
    # This simulates the np.sin pattern from your Streamlit app
    seed = random.randint(0, 1000)  # For reproducibility within one request
    rng = np.random.RandomState(seed)
    
    one_month = np.clip(np.sin(np.linspace(0, 6, months)) * 0.4 + 0.3 + rng.normal(0, 0.05, months), 0, 1).tolist()
    three_month = np.clip(np.sin(np.linspace(0.5, 6.5, months)) * 0.35 + 0.4 + rng.normal(0, 0.05, months), 0, 1).tolist()
    six_month = np.clip(np.sin(np.linspace(1, 7, months)) * 0.3 + 0.5 + rng.normal(0, 0.05, months), 0, 1).tolist()
    
    return RecessionProbabilities(
        dates=dates,
        one_month=one_month,
        three_month=three_month,
        six_month=six_month
    )

# Get Current Recession Prediction
@app.get("/api/current-prediction", response_model=RecessionPrediction)
async def get_current_prediction():
    # This would normally be the latest prediction from your model
    # For now, hardcoding similar to your dashboard metrics
    return RecessionPrediction(
        one_month=0.38,
        three_month=0.51,
        six_month=0.65,
        updated_at=datetime.now().isoformat()
    )

# Custom Prediction endpoint
@app.post("/api/custom-prediction", response_model=RecessionPrediction)
async def create_custom_prediction(request: CustomPredictionRequest):
    # This would normally call your model with the custom inputs
    # For now, implementing a simplified version of your predict_custom_recession function
    
    inputs = request.indicators
    
    # Similar weights as in your Streamlit implementation
    weights = {
        "1-Year Rate": 0.05,
        "3-Month Rate": 0.05,
        "6-Month Rate": 0.05,
        "CPI": 0.15,
        "Industrial Production": -0.15,
        "10-Year Rate": -0.1,
        "Share Price": -0.1,
        "Unemployment Rate": 0.2,
        "PPI": 0.1,
        "OECD CLI Index": -0.15,
        "CSI Index": -0.1
    }
    
    # Normalize input values
    normalized = {}
    for key in weights.keys():
        if key in inputs:
            # Simple normalization based on typical ranges
            if "Rate" in key or "CPI" in key or "PPI" in key:
                normalized[key] = inputs[key] / 10  # Rates typically 0-10%
            elif key == "Unemployment Rate":
                normalized[key] = inputs[key] / 20  # Unemployment typically 0-20%
            elif key == "Industrial Production":
                normalized[key] = (inputs[key] - 100) / 20  # Centered around 100
            elif key == "Share Price":
                normalized[key] = (inputs[key] - 4000) / 1000  # Normalized around 4000
            elif key == "OECD CLI Index":
                normalized[key] = (inputs[key] - 100) / 10  # Centered around 100
            elif key == "CSI Index":
                normalized[key] = (inputs[key] - 80) / 30  # Centered around 80
            else:
                normalized[key] = inputs[key] / 100  # Default normalization
        else:
            # Default values if not provided
            normalized[key] = 0
    
    # Calculate weighted sum
    weighted_sum = sum(weights[key] * normalized[key] for key in weights)
    
    # Check for yield curve inversion
    if "3-Month Rate" in inputs and "10-Year Rate" in inputs:
        if inputs["3-Month Rate"] > inputs["10-Year Rate"]:
            weighted_sum += 0.2  # Add recession probability for inverted yield curve
    
    # Convert to probability using sigmoid function
    import math
    base_prob = 1 / (1 + math.exp(-weighted_sum * 5))
    
    # Create slightly different probabilities for different time horizons
    return RecessionPrediction(
        one_month=min(max(base_prob, 0), 1),
        three_month=min(max(base_prob + 0.1, 0), 1),
        six_month=min(max(base_prob + 0.2, 0), 1),
        updated_at=datetime.now().isoformat()
    )

# For development testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
