import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor

import os
import requests
import dotenv
import warnings
warnings.filterwarnings("ignore")
dotenv.load_dotenv()


def fetch_latest_fred_value(series_id):
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED API key not found. Set FRED_API_KEY in your environment or pass as argument.")
    start_date="1967-02-01"
    
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch {series_id}: {response.status_code}")
        return
    data = response.json().get("observations", [])
    df = pd.DataFrame(data)
    if "date" not in df or "value" not in df:
        print(f"Series {series_id} missing required columns.")
        return
    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    
    return df


def fetch_and_combine_fred_series(output_path=None):
    data_labels = [
        "A939RX0Q048SBEA",
        "CPIAUCSL",
        "DTB1YR",
        "DTB3",
        "DTB6",
        "INDPRO",
        "IRLTLT01USM156N",
        "PCU3312103312100",
        "RECPROUSM156N",
        "SPASTT01USM661N",
        "UMCSENT",
        "UNRATE",
        "USALOLITOAASTSAM"
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fetch_latest_fred_value, data_labels))
    data_frames = dict(zip(data_labels, results))
    
    date_col='date'
    value_col='value'
    
    dataset = data_frames['RECPROUSM156N'].copy()
    dataset.rename(columns={value_col: 'recession_probability'}, inplace=True)

    ###### average monthly
    df = data_frames["DTB6"].copy()
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')  # convert to float
    
    # Create a new column representing the first day of the month
    df['month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    
    # Group by month and calculate average
    monthly_avg = df.groupby('month')[value_col].mean().reset_index()
    
    # Rename columns to match original format
    monthly_avg.rename(columns={'month': date_col}, inplace=True)
    data_frames["DTB6"] = monthly_avg
    
    # Create new columns by shifting recession_probability column
    dataset["1_month_recession_probability"] = dataset["recession_probability"].shift(-1)
    dataset["3_month_recession_probability"] = dataset["recession_probability"].shift(-3)
    dataset["6_month_recession_probability"] = dataset["recession_probability"].shift(-6)
    
    ## rates
    dataset_1yr_bond = data_frames["DTB1YR"].copy()
    dataset_3m_bill = data_frames["DTB3"].copy()
    dataset_6m_bill = data_frames["DTB6"].copy()
    
    dataset_1yr_bond = dataset_1yr_bond.rename(columns={"value": "1_year_rate"})
    dataset_3m_bill = dataset_3m_bill.rename(columns={"value": "3_months_rate"})
    dataset_6m_bill = dataset_6m_bill.rename(columns={"value": "6_months_rate"})
    
    dataset = dataset.merge(dataset_1yr_bond, on="date", how="left")
    dataset = dataset.merge(dataset_3m_bill, on="date", how="left") 
    dataset = dataset.merge(dataset_6m_bill, on="date", how="left")
    
    ## other indicators
    dataset_CPI = data_frames["CPIAUCSL"].copy()
    dataset_CPI = dataset_CPI.rename(columns={"value": "CPI"})
    dataset = dataset.merge(dataset_CPI, on="date", how="left")
    
    dataset_INDPRO = data_frames["INDPRO"].copy()
    dataset_INDPRO = dataset_INDPRO.rename(columns={"value": "INDPRO"})
    dataset = dataset.merge(dataset_INDPRO, on="date", how="left")
    
    dataset_10yr = data_frames["IRLTLT01USM156N"].copy()
    dataset_10yr = dataset_10yr.rename(columns={"value": "10_year_rate"})
    dataset = dataset.merge(dataset_10yr, on="date", how="left")
    
    dataset_share_price = data_frames["SPASTT01USM661N"].copy()
    dataset_share_price = dataset_share_price.rename(columns={"value": "share_price"})
    dataset = dataset.merge(dataset_share_price, on="date", how="left")
    
    dataset_unemployment = data_frames["UNRATE"].copy()
    dataset_unemployment = dataset_unemployment.rename(columns={"value": "unemployment_rate"})
    dataset = dataset.merge(dataset_unemployment, on="date", how="left")
    
    dataset_PPI = data_frames["PCU3312103312100"].copy()
    dataset_PPI = dataset_PPI.rename(columns={"value": "PPI"})
    dataset = dataset.merge(dataset_PPI, on="date", how="left")
    
    dataset_OECD = data_frames["USALOLITOAASTSAM"].copy()
    dataset_OECD = dataset_OECD.rename(columns={"value": "OECD_CLI_index"})
    dataset = dataset.merge(dataset_OECD, on="date", how="left")
    
    dataset_CSI = data_frames["UMCSENT"].copy()
    dataset_CSI = dataset_CSI.rename(columns={"value": "CSI_index"})
    # Replace '.' with NaN and convert column to numeric
    dataset_CSI["CSI_index"] = dataset_CSI["CSI_index"].replace('.', np.nan)
    dataset_CSI["CSI_index"] = pd.to_numeric(dataset_CSI["CSI_index"], errors='coerce')
    # Extract year and month
    dataset_CSI["date"] = pd.to_datetime(dataset_CSI["date"])
    dataset_CSI["year"] = dataset_CSI["date"].dt.year
    dataset_CSI["month"] = dataset_CSI["date"].dt.month
    # Fill missing values within same year and month
    dataset_CSI["CSI_index"] = dataset_CSI.groupby(["year"])["CSI_index"].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    dataset_CSI.drop(columns=["year", "month"], inplace=True)
    dataset = dataset.merge(dataset_CSI, on="date", how="left")
    
    ## business quater
    dataset["Business_Quarter"] = "Q" + dataset["date"].dt.quarter.astype(str)
    
    ## month and country
    dataset["Month"] = dataset["date"].dt.strftime("%B")
    dataset["Country"] = "USA"
    
    ## gdp per capita
    dataset_GDP = data_frames["A939RX0Q048SBEA"].copy()
    dataset_GDP = dataset_GDP.rename(columns={"value": "gdp_per_capita"})
    # If date is already index, make sure it's datetime
    dataset_GDP['date'] = pd.to_datetime(dataset_GDP['date'])
    dataset_GDP.set_index('date', inplace=True)
    dataset_GDP["gdp_per_capita"] = dataset_GDP["gdp_per_capita"].astype(float)
    # Resample to monthly and interpolate
    gdp_monthly_df = dataset_GDP.resample('MS').interpolate(method='linear')
    # Reset index if you want date as a column
    gdp_monthly_df = gdp_monthly_df.reset_index()
    dataset = dataset.merge(gdp_monthly_df, on="date", how="left")
    
    if output_path:
        dataset.to_csv(output_path, index=False)
        print(f"Combined dataset saved to {output_path}")
        
    return dataset

    
if __name__ == "__main__":
    fetch_and_combine_fred_series()