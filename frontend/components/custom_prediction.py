import streamlit as st
import pandas as pd
import numpy as np
from frontend.utils.data_loader import fetch_treasury_yields, load_data
from frontend.utils.visualizations import plot_custom_prediction

def display_custom_prediction():
    """Display the custom prediction page where users can adjust economic indicators"""
    
    # Initialize session state if not already done
    if 'custom_prediction_generated' not in st.session_state:
        st.session_state.custom_prediction_generated = False
    
    st.markdown("<h1 class='main-header'>Custom Recession Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Adjust economic indicators to create custom recession prediction scenarios.</p>", unsafe_allow_html=True)
    
    # Load current data for reference
    _, current_indicators, using_real_data = load_data()
    
    # Create a dictionary to store user inputs
    user_inputs = {}
    
    st.markdown("<h2 class='sub-header'>Treasury Yield Curve</h2>", unsafe_allow_html=True)
    st.markdown("<p>Adjust the Treasury yields to create different yield curve scenarios.</p>", unsafe_allow_html=True)
    
    # Create columns for the Treasury yield inputs
    cols = st.columns(4)
    
    # Treasury yield inputs
    treasury_labels = [
        '1-Month Rate', '3-Month Rate', '6-Month Rate', '1-Year Rate',
        '2-Year Rate', '5-Year Rate', '10-Year Rate', '30-Year Rate'
    ]
    
    for i, label in enumerate(treasury_labels):
        col_idx = i % 4
        with cols[col_idx]:
            default_value = current_indicators.get(label, 0.0)
            user_inputs[label] = st.number_input(
                f"{label} (%)",
                min_value=0.0,
                max_value=15.0,
                value=float(default_value),
                step=0.1,
                key=f"treasury_{label}",
                help=f"Current value: {default_value:.2f}%"
            )
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Other Economic Indicators</h2>", unsafe_allow_html=True)
    st.markdown("<p>Adjust other economic indicators that influence recession probability.</p>", unsafe_allow_html=True)
    
    # Create columns for other economic indicators
    cols = st.columns(3)
    
    # Economic indicator inputs
    econ_indicators = {
        'CPI': {'label': 'Consumer Price Index (% change)', 'min': -2.0, 'max': 15.0, 'step': 0.1},
        'Industrial Production': {'label': 'Industrial Production Index', 'min': 80.0, 'max': 120.0, 'step': 0.5},
        'Unemployment Rate': {'label': 'Unemployment Rate (%)', 'min': 2.0, 'max': 15.0, 'step': 0.1},
        'Share Price': {'label': 'S&P 500 Index', 'min': 3000.0, 'max': 6000.0, 'step': 10.0},
        'PPI': {'label': 'Producer Price Index (% change)', 'min': -2.0, 'max': 15.0, 'step': 0.1},
        'CSI Index': {'label': 'Consumer Sentiment Index', 'min': 50.0, 'max': 120.0, 'step': 0.5}
    }
    
    i = 0
    for key, config in econ_indicators.items():
        with cols[i % 3]:
            default_value = current_indicators.get(key, 0.0)
            user_inputs[key] = st.number_input(
                config['label'],
                min_value=config['min'],
                max_value=config['max'],
                value=float(default_value),
                step=config['step'],
                key=f"econ_{key}",
                help=f"Current value: {default_value:.2f}"
            )
        i += 1
    
    # Generate prediction button
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Generate Prediction", type="primary", key="generate_prediction"):
            st.session_state.custom_prediction_generated = True
    
    # Reset button
    with col2:
        if st.button("Reset to Current Values", key="reset_values"):
            st.experimental_rerun()
    
    # Show results if prediction was generated
    if st.session_state.get('custom_prediction_generated', False):
        display_custom_prediction_results(user_inputs)

def display_custom_prediction_results(user_inputs):
    """
    Display the results of the custom prediction
    
    Args:
        user_inputs (dict): User-adjusted economic indicators
    """
    st.markdown("<h2 class='sub-header'>Custom Prediction Results</h2>", unsafe_allow_html=True)
    
    # Use the actual model to make predictions
    from frontend.utils.model_utils import load_stacking_model, prepare_features, predict_recession_probability
    
    # Try to load the stacking model
    model = load_stacking_model()
    
    if model is None:
        # Fall back to simulation if model can't be loaded
        st.warning("Could not load prediction model. Using simulation instead.")
        
        # This is a simplified simulation as fallback
        def simulate_prediction(inputs):
            # Simple heuristic based on the yield curve inversion
            short_term = (inputs.get('1-Month Rate', 0) + inputs.get('3-Month Rate', 0)) / 2
            long_term = (inputs.get('10-Year Rate', 0) + inputs.get('30-Year Rate', 0)) / 2
            
            # Yield curve inversion factor (more negative = higher probability)
            inversion = long_term - short_term
            
            # Economic health factor
            economic_health = (
                inputs.get('Industrial Production', 100) / 100 * 0.2 +
                (10 - inputs.get('Unemployment Rate', 5)) / 10 * 0.3 +
                inputs.get('CSI Index', 80) / 100 * 0.2 +
                inputs.get('Share Price', 4500) / 5000 * 0.3
            )
            
            # Inflation pressure
            inflation_pressure = (
                inputs.get('CPI', 2) / 10 * 0.6 +
                inputs.get('PPI', 2) / 10 * 0.4
            )
            
            # Calculate probabilities
            base_prob = 0.3 - (inversion * 0.15) - (economic_health * 0.4) + (inflation_pressure * 0.3)
            
            # Ensure probabilities are between 0 and 1
            one_month = np.clip(base_prob, 0, 1)
            three_month = np.clip(base_prob + 0.1, 0, 1)
            six_month = np.clip(base_prob + 0.2, 0, 1)
            
            return {
                '1 Month': one_month,
                '3 Month': three_month, 
                '6 Month': six_month
            }
        
        prediction = simulate_prediction(user_inputs)
    else:
        # Use the real model for prediction
        # 1. Prepare features in the format expected by the model
        features = prepare_features(user_inputs)
        
        # 2. Make the prediction
        prediction = predict_recession_probability(model, features)
    
    # Get the prediction
    prediction = simulate_prediction(user_inputs)
    
    # Display probability metrics
    cols = st.columns(3)
    for i, (period, prob) in enumerate(prediction.items()):
        with cols[i]:
            st.metric(
                label=f"{period} Recession Probability",
                value=f"{prob:.1%}",
                delta=None
            )
    
    # Plot the custom prediction yield curve
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3>Yield Curve Analysis</h3>", unsafe_allow_html=True)
    
    # Extract treasury yields from user inputs
    treasury_yields = {k: v for k, v in user_inputs.items() if 'Rate' in k}
    
    # Plot the yield curve
    fig = plot_custom_prediction(treasury_yields, prediction)
    st.plotly_chart(fig, use_container_width=True)
    
    # Provide analysis
    st.markdown("<h3>Analysis</h3>", unsafe_allow_html=True)
    
    short_term = (user_inputs.get('1-Month Rate', 0) + user_inputs.get('3-Month Rate', 0)) / 2
    long_term = (user_inputs.get('10-Year Rate', 0) + user_inputs.get('30-Year Rate', 0)) / 2
    spread = long_term - short_term
    
    if spread < -0.5:
        risk_level = "High"
        analysis = "The yield curve is significantly inverted, indicating a high risk of recession within the next 12 months."
    elif spread < 0:
        risk_level = "Elevated"
        analysis = "The yield curve shows inversion, suggesting an elevated risk of recession in the near term."
    elif spread < 0.5:
        risk_level = "Moderate"
        analysis = "The yield curve is relatively flat, indicating some risk of economic slowdown."
    else:
        risk_level = "Low"
        analysis = "The yield curve has a healthy positive slope, suggesting low recession risk in the near term."
    
    st.info(f"**Risk Assessment: {risk_level}**\n\n{analysis}")
    
    # Disclaimer
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.caption(
        "Disclaimer: This is a simplified simulation for educational purposes only. "
        "Actual recession predictions require comprehensive models and analysis."
    )
