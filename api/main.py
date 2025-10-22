import os, sys, time
import requests
import dotenv
import asyncio
import threading, concurrent.futures
from typing import Dict, List
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dotenv.load_dotenv()

import pandas as pd
import numpy as np
from pydantic import BaseModel
from ML_pipe import time_series_prediction, time_series_feature_eng, regresstion_feature_engineering, regression_prediction
from data_collection import fetch_and_combine_fred_series

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


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

class HistoricalEconomicData(BaseModel):
    dates: List[str]
    cpi: List[float]
    ppi: List[float]
    industrial_production: List[float]
    unemployment_rate: List[float]
    share_price: List[float]
    gdp_per_capita: List[float]
    oecd_cli_index: List[float]
    csi_index: List[float]
    ten_year_rate: List[float]
    three_months_rate: List[float]
    six_months_rate: List[float]
    one_year_rate: List[float]

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

# NEW: Financial advice I/O models
class FinancialAdviceRequest(BaseModel):
    forecasted_indicators: Dict[str, float]
    recession_probabilities: Dict[str, float]

class FinancialAdviceResponse(BaseModel):
    advice: str
    generated_at: str
    
FRED_API_KEY = os.getenv("FRED_API_KEY")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

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
ts_prediction = None
ts_fe_data = None
fetched_data = None


def run_ml_pipe():
    global fetched_data, ts_fe_data, ts_prediction, latest_predictions, fe_data
    
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3] ,"Running ML pipeline...")
    fetched_data = fetch_and_combine_fred_series()
    print(f"Data fetched from {fetched_data.iloc[0]['date']} to {fetched_data.iloc[-1]['date']}, total {len(fetched_data)} records.")
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3],"Fetched latest FRED data.")
    
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Running time series prediction...")
    ts_fe_data = time_series_feature_eng(fetched_data)
    ts_prediction = time_series_prediction(ts_fe_data)
    print("Time series predictions:", ts_prediction.T)
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3],"Time series prediction completed.")
    
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Running regression prediction...")
    fe_data = regresstion_feature_engineering(ts_fe_data, ts_prediction)
    try:
        base, one_month, three_month, six_month= regression_prediction(fe_data)
        latest_predictions = {
            "base_pred": base/100,
            "one_month": one_month/100,
            "three_month": three_month/100,
            "six_month": six_month/100,
            "updated_at": datetime.now().isoformat()
        }
        print(f"Base: {base}, 1m: {one_month}, 3m: {three_month}, 6m: {six_month}")

    except Exception as e:
        print("Error in ML pipeline:", e)
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3],"Regression prediction completed.")


def run_ml_pipeline_periodically():
    global latest_predictions, yields, indicators, recession_data, ts_prediction, ts_fe_data, fetched_data
    while True:
        try:
            ## ML PIPELINE
            start = datetime.now()
            run_ml_pipe()
            
            ## Fetch Treasury Yields and Economic Indicators
            print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Fetching Treasury Yields and Economic Indicators...")
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
            print("Updated yields:")
            print("Updated indicators:")
            print(datetime.now().strftime("%H:%M:%S.%f")[:-3],"Fetch completed.")
            
            ## recession dataset
            full_df = fetched_data.copy().reset_index(drop=True)
            full_df = full_df.sort_values('date').reset_index(drop=True)
            
            # Replace NaN, inf, -inf with 0.0
            full_df = full_df.replace([np.nan, np.inf, -np.inf], 0.0)
            
            recession_data = full_df[["date", "recession_probability", "1_month_recession_probability", 
                                      "3_month_recession_probability", "6_month_recession_probability"]].copy() 
            print(datetime.now().strftime("%H:%M:%S.%f")[:-3], "Updated recession data:")
            print("pipeline duration:", datetime.now() - start)
            
            time.sleep(6 * 60 * 60)  # 6 hours
        except Exception as e:
            print(f"Error in ML pipeline: {e}")
            print("Continuing with next iteration...")
            time.sleep(60)  # Wait 1 minute before retrying on error


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

# Startup event to start ML pipeline
@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=run_ml_pipeline_periodically, daemon=True)
    thread.start()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to RecessionRadar API"}




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

# Get Historical Economic Indicators Data
@app.get("/api/historical-economic-data", response_model=HistoricalEconomicData)
async def get_historical_economic_data():
    try:
        # Read the CSV file directly
        import pandas as pd
        import os
        
        # Get the correct path to the CSV file
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'recession_probability.csv')
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Data file not found")
        
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.fillna(0)  # Replace NaN with 0
        
        return HistoricalEconomicData(
            dates=[d.strftime("%Y-%m-%d") for d in df["date"]],
            cpi=df["CPI"].tolist(),
            ppi=df["PPI"].tolist(),
            industrial_production=df["INDPRO"].tolist(),
            unemployment_rate=df["unemployment_rate"].tolist(),
            share_price=df["share_price"].tolist(),
            gdp_per_capita=df["gdp_per_capita"].tolist(),
            oecd_cli_index=df["OECD_CLI_index"].tolist(),
            csi_index=df["CSI_index"].tolist(),
            ten_year_rate=df["10_year_rate"].tolist(),
            three_months_rate=df["3_months_rate"].tolist(),
            six_months_rate=df["6_months_rate"].tolist(),
            one_year_rate=df["1_year_rate"].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading economic data: {str(e)}")


# Get Current Recession Prediction
@app.get("/api/current-prediction", response_model=RecessionPrediction)
async def get_current_prediction():
    return RecessionPrediction(**latest_predictions)


# Custom Prediction endpoint
@app.post("/api/custom-prediction", response_model=RecessionPrediction)
async def create_custom_prediction(request: CustomPredictionRequest):
    global ts_fe_data, fetched_data
    
    inputs = request.indicators
    new_row = {
        'date': datetime.now(),
        '1_year_rate': inputs['1-Year Rate'],
        '3_months_rate': inputs['3-Month Rate'],
        '6_months_rate': inputs['6-Month Rate'],
        'CPI': inputs['CPI'],
        'INDPRO': inputs['Industrial Production'],
        '10_year_rate': inputs['10-Year Rate'],
        'share_price': inputs['Share Price'],
        'unemployment_rate': inputs['Unemployment Rate'],
        'PPI': inputs['PPI'],
        'OECD_CLI_index': inputs['OECD CLI Index'],
        'CSI_index': inputs['CSI Index'],
        'gdp_per_capita': ts_prediction['gdp_per_capita'].iloc[-1]
    }
    fetched_data_ = fetched_data.copy()
    fetched_data_ = pd.concat([fetched_data_, pd.DataFrame([new_row])], ignore_index=True)
    inputs = time_series_feature_eng(fetched_data_).iloc[[-1]]
    custom_data = regresstion_feature_engineering(ts_fe_data, inputs)
    print(custom_data.T)
    base_pred, one_month, three_month, six_month = regression_prediction(custom_data)
    print(f"Custom Prediction - Base: {base_pred}, 1m: {one_month}, 3m: {three_month}, 6m: {six_month}")
    return RecessionPrediction(
        base_pred = min(max(base_pred/100, 0), 1),
        one_month=min(max(one_month/100, 0), 1),
        three_month=min(max(three_month/100, 0), 1),
        six_month=min(max(six_month/100, 0), 1),
        updated_at=datetime.now().isoformat()
    )


# Helper to build prompt for OpenAI
def build_advice_prompt(forecasted_indicators: Dict[str, float],
                        recession_probabilities: Dict[str, float]) -> str:
    header = (
        "You are an intelligent financial advisor on US markets. "
        "Provide a concise, actionable, high-level market and portfolio guidance based on the inputs. "
        "Cover equity (stocks), fixed income (bonds, duration/credit tilt), and cash/alternatives. "
        "Mention risk management and diversification. "
        "End every answer with exactly: This is not Buy, Hold or Sell advice."
    )
    parts = ["Forecasted financial indicators:"]
    for k, v in forecasted_indicators.items():
        parts.append(f"- {k}: {v}")
    parts.append("\nRecession probabilities:")
    for k, v in recession_probabilities.items():
        parts.append(f"- {k}: {v}")
    parts.append(
        "\nGiven the above, provide the guidance now. Keep it under 250 words."
    )
    return header + "\n\n" + "\n".join(parts)

# NEW: Endpoint to get financial advice from server-side predictions (no body)
@app.get("/api/financial-advice", response_model=FinancialAdviceResponse)
async def get_financial_advice():
    global ts_prediction, latest_predictions, indicators, yields

    if not OPEN_AI_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    # Ensure ML outputs are available
    if ts_prediction is None or latest_predictions.get("base_pred") is None:
        raise HTTPException(status_code=503, detail="Forecasts not available yet. Try again later.")

    # Build forecasted indicators using the in-memory ts_prediction, indicators and yields
    forecasted_indicators = {}

    # Prefer explicit latest fetched indicators / yields (these are simple scalars)
    try:
        # copy known indicator keys
        for k in ["CPI", "PPI", "Industrial Production", "Share Price", "Unemployment Rate", "OECD CLI Index", "CSI Index"]:
            if k in indicators and indicators[k] is not None:
                forecasted_indicators[k] = float(indicators[k])
        # yields (rates)
        for k in ["3-Month Rate", "6-Month Rate", "1-Year Rate", "10-Year Rate"]:
            if k in yields and yields[k] is not None:
                forecasted_indicators[k] = float(yields[k])
    except Exception:
        pass

    # Fallback: try to read numeric columns from ts_prediction last row
    last_row = None
    try:
        if hasattr(ts_prediction, "iloc"):
            last_row = ts_prediction.iloc[-1]
        elif isinstance(ts_prediction, dict):
            last_row = pd.Series(ts_prediction)
    except Exception:
        last_row = None

    if last_row is not None:
        mapping_candidates = {
            "CPI": ["CPI", "cpi"],
            "PPI": ["PPI", "ppi"],
            "Industrial Production": ["INDPRO", "industrial_production", "indpro"],
            "Share Price": ["share_price", "SP500", "sharePrice"],
            "Unemployment Rate": ["unemployment_rate", "UNRATE", "unemploymentRate"],
            "3-Month Rate": ["3_months_rate", "3_months_rate", "3_month_rate", "DTB3"],
            "6-Month Rate": ["6_months_rate", "6_months_rate", "6_month_rate", "DTB6"],
            "1-Year Rate": ["1_year_rate", "1_year_rate", "DTB1YR"],
            "10-Year Rate": ["10_year_rate", "10_year_rate", "DGS10"]
        }
        for target, candidates in mapping_candidates.items():
            if target in forecasted_indicators:
                continue
            for col in candidates:
                if col in last_row and pd.notna(last_row[col]):
                    try:
                        forecasted_indicators[target] = float(last_row[col])
                        break
                    except Exception:
                        continue

    # Final guard: ensure at least some indicators exist
    if not forecasted_indicators:
        raise HTTPException(status_code=503, detail="No forecasted indicators available to generate advice.")

    # Build recession probabilities from latest_predictions
    recession_probabilities = {
        "base_pred": latest_predictions.get("base_pred"),
        "one_month": latest_predictions.get("one_month"),
        "three_month": latest_predictions.get("three_month"),
        "six_month": latest_predictions.get("six_month")
    }

    prompt = build_advice_prompt(forecasted_indicators, recession_probabilities)

    def _call_openai():
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPEN_AI_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 600,
        }
        print("Prompt:\n\n", prompt)
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    try:
        data = await asyncio.to_thread(_call_openai)
        advice = (data.get("choices", [{}])[0]
                     .get("message", {})
                     .get("content", "")
                     .strip())
        if "This is not Buy, Hold or Sell advice." not in advice:
            advice = advice.rstrip() + "\n\nThis is not Buy, Hold or Sell advice."
        return FinancialAdviceResponse(
            advice=advice,
            generated_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate advice: {e}")


# Manual trigger endpoint for ML pipeline.
# Runs run_ml_pipe in a thread and returns success after completion.
@app.post("/api/run-ml-pipeline")
async def run_ml_pipeline_manual():
    try:
        # Run blocking pipeline in a thread so event loop is not blocked
        await asyncio.to_thread(run_ml_pipe)
        return {"status": "success", "message": "ML pipeline executed successfully", "updated_at": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running ML pipeline: {str(e)}")


# For development testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
