# RecessionRadar: React Frontend + FastAPI Backend

This folder contains the React frontend for the RecessionRadar application. The application provides recession probability forecasts based on economic indicators and treasury yield data.

## Setup Instructions

1. Make sure you have Node.js installed (version 14+)
2. Install dependencies: `npm install`
3. Start the development server: `npm start`

## Features

- Dashboard with recession probability metrics
- Historical recession probability charts
- Treasury yield curve visualization
- Custom prediction tool for scenario analysis
- Information page explaining methodology

## Project Structure

- `src/components/`: Reusable UI components
- `src/pages/`: Main application pages
- `src/services/`: API service layer

## Dependencies

- React 18
- Material UI for styling
- Chart.js for data visualization
- Axios for API requests
- React Router for navigation

## API Integration

The frontend communicates with the FastAPI backend at `/api` endpoints. See the API documentation for details on available endpoints and data formats.

## Development

During development, the React app will proxy API requests to the backend server running on port 8000. In production, both the frontend and backend should be served from the same domain.
