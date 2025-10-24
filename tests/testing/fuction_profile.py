import cProfile, pstats, io
import time, os, dotenv, sys
import requests
import numpy as np
import pandas as pd
import concurrent.futures
from datetime import datetime
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import from api.data_collection
data_collection = importlib.import_module("api.data_collection")
fetch_and_combine_fred_series = getattr(data_collection, "fetch_and_combine_fred_series")

# Import from api.ML_pipe
ml_pipe = importlib.import_module("api.ML_pipe")
time_series_feature_eng = getattr(ml_pipe, "time_series_feature_eng")
time_series_prediction = getattr(ml_pipe, "time_series_prediction")
regresstion_feature_engineering = getattr(ml_pipe, "regresstion_feature_engineering")
regression_prediction = getattr(ml_pipe, "regression_prediction")

lgbm_wrapper = importlib.import_module("api.lgbm_wrapper")


dotenv.load_dotenv()
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
fe_data = None

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


def run_ml_pipeline_periodically():
    global latest_predictions, yields, indicators, recession_data, ts_prediction, ts_fe_data, fetched_data, fe_data
    while True:
        try:
            ## ML PIPELINE
            start = datetime.now()
            fetched_data = fetch_and_combine_fred_series()
            ts_fe_data = time_series_feature_eng(fetched_data)
            ts_prediction = time_series_prediction(ts_fe_data)
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
            except Exception as e:
                print("Error in ML pipeline:", e)

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
            
            ## recession dataset
            full_df = fetched_data.copy().reset_index(drop=True)
            full_df = full_df.sort_values('date').reset_index(drop=True)
            
            # Replace NaN, inf, -inf with 0.0
            full_df = full_df.replace([np.nan, np.inf, -np.inf], 0.0)
            
            recession_data = full_df[["date", "recession_probability", "1_month_recession_probability", 
                                      "3_month_recession_probability", "6_month_recession_probability"]].copy() 
            
            # time.sleep(300)  # 5 minutes
            return
        except Exception as e:
            print(f"Error in ML pipeline: {e}")
            raise e
            print("Continuing with next iteration...")
            # time.sleep(60)  # Wait 1 minute before retrying on error



def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()

    func(*args, **kwargs)

    profiler.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(20)
    print(s.getvalue())
    
if __name__ == "__main__":
    print("Profiling the ML pipeline function...")
    profile_function(run_ml_pipeline_periodically)
    print()
    
    print("Profiling the time series feature engineering function...")
    profile_function(time_series_feature_eng, fetched_data)
    print()
    
    print("Profiling the time series prediction function...")
    profile_function(time_series_prediction, ts_fe_data)
    print()
    
    print("Profiling the regression feature engineering function...")
    profile_function(regresstion_feature_engineering, ts_fe_data, ts_prediction)
    print()
    
    print("Profiling the regression prediction function...")
    profile_function(regression_prediction, fe_data)
    print()
