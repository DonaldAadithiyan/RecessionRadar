# US Recession Forecasting

## Project Overview

Forecasting future recessions can help policy makers and businesses take preventive actions or prepare for economic downturns. In this project, we aim to use historical economic indicators to predict the probability of a recession in the United States occurring **1, 3, or 6 months ahead**.

Understanding economic downturns and their leading indicators can help:

- **Government** agencies design better policies,
- **Investors** manage risks more effectively,
- **Businesses** plan for supply, demand, and workforce accordingly.

## ⚙️ Installation & Execution Guide

This repository contains two parts:

- Backend (Python API & ML code) in the `api/` folder
- Frontend (React) in the `react-frontend/` folder

Follow the instructions below on Windows (PowerShell). Adjust commands for other shells/OS.

---

### Backend (Python)

Prerequisites:

- Python 3.10+ (3.12 tested in this repo)
- Git (optional)

1. Open PowerShell and go to the project root:
   ```cd "path\to\project_dir"```

2. Create and activate a virtual environment:

  ```powershell
  python -m venv .\venv
  .\venv\Scripts\Activate
  ```

1. Install backend dependencies:

   ```powershell
   pip install --upgrade pip
   pip install -r .\api\requirements.txt
   ```

2. Run the API / backend:
   - If the project uses Uvicorn / FastAPI (recommended):
     ```uvicorn api.main:app --reload --port 8000```
   - Otherwise run the entry script:
     ```python .\api\main.py```

---

### Frontend (React)

Prerequisites:

- Node.js (16+) and npm or yarn

1. Open PowerShell and go to the frontend folder:

   ```powershell
    cd "path\to\project_dir\react-frontend"
    ```

2. Install packages:

   ```npm install```

   (or)

   ```yarn install```

3. Configure API endpoint (optional):
   - By default the frontend calls the API URL defined in `src/services/api.js`.
   - To override via env var (React): in PowerShell before starting:
     ```$env:REACT_APP_API_URL = "http://localhost:8000"```

4. Start dev server:
   - npm start
   - Open http://localhost:3000 in your browser

5. Build for production:
   ```npm run build```
   - The optimized build is created in `react-frontend/build`.

---

## Common Troubleshooting

- ModuleNotFoundError: ensure you run Python commands from project root or set PYTHONPATH to project root.
- Missing dependencies: ensure you installed `api/requirements.txt` inside the activated venv.
- Frontend cannot reach backend: confirm backend is running and CORS/port match; set REACT_APP_API_URL if needed.

---

## Project Structure (summary)

- api/                Backend code, ML pipeline, API entrypoint
- react-frontend/     React app for dashboard and visualizations

---
