# RecessionRadar API

FastAPI backend for RecessionRadar — provides economic data and recession probability predictions.

---

## Requirements & env vars

- Python 3.10+ (3.12 tested)
- FRED API key set in environment variable `FRED_API_KEY` (required for live FRED data)

Example (PowerShell):

```powershell
$env:FRED_API_KEY = "your_fred_api_key_here"
```

---

## Setup (from project root)

Open PowerShell at project root `D:\edu\sem 5\DataScience Engineering Project\RecessionRadar`:

```powershell
python -m venv .\venv
.\venv\Scripts\Activate
pip install --upgrade pip
pip install -r .\requirements.txt
```

If you get `ModuleNotFoundError: No module named 'api'`, run commands from project root or set PYTHONPATH:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

---

## Run (development)

From project root:

```powershell
uvicorn api.main:app --reload --port 8000
```

Or (inside api folder):

```powershell
python .\main.py
```

Docs available when running:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Endpoints (implemented in api/main.py)

- GET /  
  - Root welcome message.

- GET /api/treasury-yields  
  - Returns latest treasury yields (dictionary) and updated_at timestamp.

- GET /api/economic-indicators  
  - Returns latest economic indicators (dictionary) and updated_at timestamp.

- GET /api/recession-probabilities  
  - Returns historical recession probability series:
    - dates, base (recession_probability), one_month, three_month, six_month

- GET /api/historical-economic-data  
  - Loads CSV (data/recession_probability.csv) and returns arrays for:
    - dates, CPI, PPI, Industrial Production, Unemployment Rate, Share Price, GDP per capita, OECD CLI index, CSI index, and several rates (3mo, 6mo, 1yr, 10yr).
  - Returns 404 if CSV not found.

- GET /api/current-prediction  
  - Returns the latest prediction computed by the background ML pipeline:
    - base_pred, one_month, three_month, six_month, updated_at

- POST /api/custom-prediction  
  - Accepts JSON body with `indicators` (map of input feature names → values). Runs a quick custom prediction using current pipeline state and returns RecessionPrediction (base_pred, one_month, three_month, six_month, updated_at).
  - Example request body:

    ```json
    {
      "indicators": {
        "1-Year Rate": 4.2,
        "3-Month Rate": 5.1,
        "6-Month Rate": 4.7,
        "10-Year Rate": 3.9,
        "CPI": 0.2,
        "PPI": 0.15,
        "Industrial Production": 0.01,
        "Share Price": 4200,
        "Unemployment Rate": 3.7,
        "OECD CLI Index": 100.0,
        "CSI Index": 50.0,
        "PPI": 0.15
      }
    }
    ```

- POST /api/run-ml-pipeline  
  - Manually trigger the full ML pipeline (runs synchronously in a background thread and returns status).

Notes:

- The app starts a background thread on startup that periodically (every 6 hours) runs the ML pipeline: it fetches FRED data, computes features, runs time-series and regression predictions, and updates in-memory latest predictions and time series data.
- If `FRED_API_KEY` is missing, the pipeline will raise an error when attempting live fetches.

---

## Dev tips

- Keep the venv activated and run backend and frontend in separate terminals.
- Ensure `api/__init__.py` exists if you import `api` from scripts (not required for uvicorn).
- To test ML pipeline manually:

  ```powershell
  & .\venv\Scripts\python.exe .\notebooks\test\test.py
  ```

---

## Production

- Replace "*" CORS origins with your frontend URL.
- Persist model outputs and data to disk/database instead of keeping only in-memory state for reliability.
- Secure the API key and any endpoints (authentication, authorization) as needed.
