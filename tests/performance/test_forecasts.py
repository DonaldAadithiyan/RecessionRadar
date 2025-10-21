#!/usr/bin/env python3
"""
Test script to demonstrate the actual vs forecasted concept
This shows what the chart should look like with realistic predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Sample actual recession probabilities (like what we get from FRED)
actual_data = [
    0.06, 0.06, 0.30, 0.10, 0.06, 0.02, 0.04, 0.10, 0.24, 0.10,  # 2020-2021
    0.06, 0.011, 0.012, 0.08, 0.12, 0.44, 0.34, 0.10, 0.08, 0.28,  # 2021-2022
    0.56, 0.40, 0.16, 0.26, 0.32, 0.68, 0.012, 0.013, 0.26, 0.42,  # 2022-2023
    0.44, 0.34, 0.42, 0.56, 0.28, 0.28, 0.30, 0.60, 0.38, 0.54,   # 2023-2024
    0.014, 0.26, 0.44, 0.26, 0.08, 0.20, 0.38, 0.12, 0.18, 0.30   # 2024-2025
]

# Generate realistic forecasted values that follow economic patterns
# but are imperfect predictions (as any real model would be)
forecasted_data = []
np.random.seed(42)  # For reproducible results

for i, actual in enumerate(actual_data):
    # Base prediction with some accuracy but not perfect
    base_forecast = actual * (0.7 + np.random.normal(0, 0.3))
    
    # Add economic cycle patterns
    cycle_factor = 1 + 0.4 * np.sin(i * 0.2)
    base_forecast *= cycle_factor
    
    # Add trend - models often miss turning points
    if i > 0:
        trend = (actual_data[i] - actual_data[i-1]) * 0.5  # Dampened trend following
        base_forecast += trend
    
    # Keep within reasonable bounds
    base_forecast = max(min(base_forecast, 0.9), 0.005)
    forecasted_data.append(base_forecast)

# Generate dates
start_date = datetime(2020, 9, 1)
dates = [(start_date + timedelta(days=30*i)).strftime('%Y-%m-%d') for i in range(len(actual_data))]

# Create the comparison
print("=== ACTUAL vs FORECASTED RECESSION PROBABILITY COMPARISON ===")
print("This demonstrates what your chart should show:\n")

print(f"{'Month':<10} {'Actual %':<10} {'Forecast %':<12} {'Difference':<12} {'Assessment'}")
print("-" * 60)

for i, (date, actual, forecast) in enumerate(zip(dates, actual_data, forecasted_data)):
    month = date[:7]  # YYYY-MM format
    actual_pct = actual * 100
    forecast_pct = forecast * 100
    diff = abs(actual_pct - forecast_pct)
    
    if diff < 5:
        assessment = "Good"
    elif diff < 15:
        assessment = "Fair"
    else:
        assessment = "Poor"
    
    print(f"{month:<10} {actual_pct:<10.1f} {forecast_pct:<12.1f} {diff:<12.1f} {assessment}")

print(f"\nSummary Statistics:")
print(f"Average Actual: {np.mean(actual_data)*100:.1f}%")
print(f"Average Forecast: {np.mean(forecasted_data)*100:.1f}%")
print(f"Mean Absolute Error: {np.mean([abs(a-f)*100 for a,f in zip(actual_data, forecasted_data)]):.1f}%")
print(f"Correlation: {np.corrcoef(actual_data, forecasted_data)[0,1]:.3f}")

print("\n=== KEY INSIGHTS ===")
print("1. Forecasts should VARY significantly (not be flat)")
print("2. Forecasts should follow general trends but miss some peaks/valleys")
print("3. Model performance can be measured by MAE and correlation")
print("4. Chart should show both lines with clear differences")

# Sample JSON output format for the API
sample_output = {
    "indicator": "recession_probability",
    "months": len(dates),
    "actuals": {
        "current": {
            "dates": dates,
            "values": actual_data
        }
    },
    "forecasts": {
        "current": {
            "dates": dates,
            "values": forecasted_data
        }
    }
}

print(f"\n=== SAMPLE API OUTPUT (first 5 points) ===")
sample_small = {
    "actuals": sample_output["actuals"]["current"]["values"][:5],
    "forecasts": sample_output["forecasts"]["current"]["values"][:5],
    "dates": sample_output["actuals"]["current"]["dates"][:5]
}
print(json.dumps(sample_small, indent=2))