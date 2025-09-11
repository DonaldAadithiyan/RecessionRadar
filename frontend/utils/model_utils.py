import joblib
import pandas as pd
import numpy as np
import os
import streamlit as st

# Path to the models directory - adjust as needed
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

@st.cache_resource
def load_stacking_model():
    """
    Load the stacking ensemble model from disk
    
    Returns:
        object: The trained stacking ensemble model
    """
    model_path = os.path.join(MODELS_DIR, "stacking_ensemble_model.joblib")
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_recession_probability(model, features):
    """
    Predict recession probability using the loaded model
    
    Args:
        model: The trained model
        features (dict): Dictionary of feature values
        
    Returns:
        dict: Dictionary of recession probabilities for different time horizons
    """
    if model is None:
        return {
            '1 Month': 0.5,
            '3 Month': 0.5,
            '6 Month': 0.5
        }
    
    # Convert features dictionary to DataFrame with expected structure
    # This will need to be customized based on your model's expected input format
    try:
        # Extract features in the order expected by the model
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        probabilities = model.predict_proba(feature_df)
        
        # Get probability of recession (assuming binary classification with class 1 = recession)
        recession_prob = probabilities[0][1]
        
        # Create simulated probabilities for different horizons
        # In a real implementation, you might have separate models for each horizon
        return {
            '1 Month': recession_prob,
            '3 Month': min(recession_prob * 1.2, 1.0),  # Slightly higher for longer horizon
            '6 Month': min(recession_prob * 1.4, 1.0)   # Even higher for 6-month horizon
        }
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return {
            '1 Month': 0.5,
            '3 Month': 0.5,
            '6 Month': 0.5
        }

def prepare_features(inputs):
    """
    Transform raw inputs into features expected by the model
    
    Args:
        inputs (dict): Raw input values from the UI
        
    Returns:
        dict: Processed features ready for the model
    """
    features = {}
    
    # Extract Treasury yield curve features
    if all(k in inputs for k in ['3-Month Rate', '10-Year Rate']):
        # Calculate yield curve spread (10yr - 3mo)
        features['yield_spread_10y_3m'] = inputs['10-Year Rate'] - inputs['3-Month Rate']
    
    if all(k in inputs for k in ['2-Year Rate', '10-Year Rate']):
        # Calculate yield curve spread (10yr - 2yr)
        features['yield_spread_10y_2y'] = inputs['10-Year Rate'] - inputs['2-Year Rate']
    
    # Add other important economic indicators
    for key in ['CPI', 'Unemployment Rate', 'Industrial Production', 'Share Price']:
        if key in inputs:
            # Convert to feature name format (lowercase with underscores)
            feature_name = key.lower().replace(' ', '_')
            features[feature_name] = inputs[key]
    
    # Add any necessary transformations or interactions
    if 'CPI' in inputs and 'PPI' in inputs:
        features['inflation_pressure'] = (inputs['CPI'] + inputs['PPI']) / 2
    
    # Return the processed features
    return features
