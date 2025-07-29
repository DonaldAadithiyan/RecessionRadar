import os
import requests
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get the FRED API key from environment
API_KEY = os.getenv("FRED_API_KEY")
series_id = "UMCSENT"

print(API_KEY)

url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": series_id,
    "api_key": API_KEY,
    "file_type": "json",
    "observation_start": "1967-02-01"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    
    # Extract the 'observations' list from the response
    observations = data.get("observations", [])
    
    # Convert to DataFrame
    df = pd.DataFrame(observations)
    
    # Specify output directory and filename
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if not exists
    output_path = os.path.join(output_dir, f"{series_id}.csv")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
else:
    print(f"Error: {response.status_code} - {response.text}")
