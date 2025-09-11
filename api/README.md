# RecessionRadar API

This folder contains the FastAPI backend for the RecessionRadar application.

## Setup Instructions

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the API server:
   ```
   uvicorn main:app --reload
   ```

## API Endpoints

The API provides the following endpoints:

- `GET /api/treasury-yields`: Get current treasury yield rates
- `GET /api/economic-indicators`: Get current economic indicators
- `GET /api/recession-probabilities`: Get historical recession probabilities
- `GET /api/current-prediction`: Get current recession predictions
- `POST /api/custom-prediction`: Generate custom recession predictions based on user inputs

## Development

The API can be run independently for development purposes. For production, it should be mounted within the main FastAPI application (app.py).

## Documentation

When the server is running, API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Note

This implementation uses hardcoded data and simplified prediction logic for demonstration purposes. In a production setting, it would connect to a trained machine learning model and fetch real-time data from financial APIs.
