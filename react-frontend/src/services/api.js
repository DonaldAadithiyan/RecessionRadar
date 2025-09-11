import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Treasury Yields API
export const getTreasuryYields = async () => {
  try {
    const response = await axios.get(`${API_URL}/treasury-yields`);
    return response.data;
  } catch (error) {
    console.error('Error fetching treasury yields:', error);
    throw error;
  }
};

// Economic Indicators API
export const getEconomicIndicators = async () => {
  try {
    const response = await axios.get(`${API_URL}/economic-indicators`);
    return response.data;
  } catch (error) {
    console.error('Error fetching economic indicators:', error);
    throw error;
  }
};

// Recession Probabilities API
export const getRecessionProbabilities = async (months = 24) => {
  try {
    const response = await axios.get(`${API_URL}/recession-probabilities?months=${months}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching recession probabilities:', error);
    throw error;
  }
};

// Current Prediction API
export const getCurrentPrediction = async () => {
  try {
    const response = await axios.get(`${API_URL}/current-prediction`);
    return response.data;
  } catch (error) {
    console.error('Error fetching current prediction:', error);
    throw error;
  }
};

// Custom Prediction API
export const getCustomPrediction = async (indicators) => {
  try {
    const response = await axios.post(`${API_URL}/custom-prediction`, {
      indicators: indicators
    });
    return response.data;
  } catch (error) {
    console.error('Error getting custom prediction:', error);
    throw error;
  }
};
