import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="RecessionRadar Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #424242 !important;
    }
    .info-text {
        font-size: 1rem !important;
        color: #616161 !important;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
    }
    .warning {
        color: #ff6b6b;
        font-weight: 600;
    }
    .success {
        color: #51cf66;
        font-weight: 600;
    }
    .metrics-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.15rem 0.3rem rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metrics-card:hover {
        transform: translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

# Function to load sample data
def load_sample_data():
    # Generate sample recession probability data
    today = datetime.now()
    dates = [today - timedelta(days=30*i) for i in range(120)]
    dates.reverse()
    
    # Create DataFrame with dates as index
    df = pd.DataFrame({
        'date': dates,
        'recession_probability_1m': np.clip(np.sin(np.linspace(0, 6, 120)) * 0.4 + 0.3 + np.random.normal(0, 0.05, 120), 0, 1),
        'recession_probability_3m': np.clip(np.sin(np.linspace(0.5, 6.5, 120)) * 0.35 + 0.4 + np.random.normal(0, 0.05, 120), 0, 1),
        'recession_probability_6m': np.clip(np.sin(np.linspace(1, 7, 120)) * 0.3 + 0.5 + np.random.normal(0, 0.05, 120), 0, 1)
    })
    df.set_index('date', inplace=True)
    
    # Sample economic indicators (most recent values) - Updated based on the image
    indicators = {
        '1-Year Rate': 4.86,
        '3-Month Rate': 4.52,
        '6-Month Rate': 4.65,
        'CPI': 3.2,
        'Industrial Production': 104.5,
        '10-Year Rate': 3.81,
        'Share Price': 4512.8,
        'Unemployment Rate': 3.8,
        'PPI': 2.8,
        'OECD CLI Index': 99.2,
        'CSI Index': 82.7
    }
    
    return df, indicators

# Simulate predicting recession probability based on user inputs
def predict_custom_recession(inputs):
    # This is a simplified heuristic that simulates model prediction
    # In a real application, this would use the loaded model
    
    # Calculate weighted sum of normalized indicators
    # These weights are completely arbitrary for demonstration
    weights = {
        '1-Year Rate': 0.05,
        '3-Month Rate': 0.05,
        '6-Month Rate': 0.05,
        'CPI': 0.15,
        'Industrial Production': -0.15,
        '10-Year Rate': -0.1,
        'Share Price': -0.1,
        'Unemployment Rate': 0.2,
        'PPI': 0.1,
        'OECD CLI Index': -0.15,
        'CSI Index': -0.1
    }
    
    # Normalize input values based on typical ranges (completely made up for demo)
    normalized = {
        '1-Year Rate': (inputs['1-Year Rate'] - 0) / 8,  # assume range 0-8
        '3-Month Rate': (inputs['3-Month Rate'] - 0) / 8,  # assume range 0-8
        '6-Month Rate': (inputs['6-Month Rate'] - 0) / 8,  # assume range 0-8
        'CPI': (inputs['CPI'] - 0) / 10,  # assume range 0-10
        'Industrial Production': (inputs['Industrial Production'] - 90) / 30,  # assume range 90-120
        '10-Year Rate': (inputs['10-Year Rate'] - 0) / 8,  # assume range 0-8
        'Share Price': (inputs['Share Price'] - 3000) / 2000,  # assume range 3000-5000
        'Unemployment Rate': (inputs['Unemployment Rate'] - 3) / 7,  # assume range 3-10
        'PPI': (inputs['PPI'] - 0) / 10,  # assume range 0-10
        'OECD CLI Index': (inputs['OECD CLI Index'] - 95) / 10,  # assume range 95-105
        'CSI Index': (inputs['CSI Index'] - 60) / 40  # assume range 60-100
    }
    
    # Calculate weighted sum
    weighted_sum = sum(weights[key] * normalized[key] for key in weights)
    
    # Check for yield curve inversion (3-Month > 10-Year) - adds recession probability
    if inputs['3-Month Rate'] > inputs['10-Year Rate']:
        weighted_sum += 0.2
    
    # Convert to probability using sigmoid function
    base_prob = 1 / (1 + np.exp(-weighted_sum * 5))
    
    # Create slightly different probabilities for different time horizons
    return {
        '1-Month': min(max(base_prob, 0), 1),
        '3-Month': min(max(base_prob + 0.1, 0), 1),
        '6-Month': min(max(base_prob + 0.2, 0), 1)
    }

# Main app
def main():
    # Create sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/economic-improvement.png", width=80)
        st.title("Recession Radar")
        
        # Navigation
        page = st.radio("Navigation", ["Dashboard", "Custom Prediction", "Information"])
        
        # Disclaimer
        st.markdown("---")
        st.caption("**Disclaimer:** This application provides recession probability forecasts based on economic indicators. These forecasts are not financial advice and should not be used as the sole basis for financial decisions.")
        
        # About section
        st.markdown("---")
        st.caption("**About**")
        st.caption("RecessionRadar is a dashboard for visualizing and forecasting recession probabilities based on economic indicators.")
        
        # Version
        st.markdown("---")
        st.caption("Version 1.0 | Demo Version")
    
    # Page content
    if page == "Dashboard":
        display_dashboard()
    elif page == "Custom Prediction":
        display_custom_prediction()
    else:  # Information
        display_information()

def display_dashboard():
    st.markdown("<h1 class='main-header'>Recession Radar Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>This dashboard visualizes the probability of a recession occurring within the next 1, 3, and 6 months based on economic indicators.</p>", unsafe_allow_html=True)
    
    # Load sample data
    df, indicators = load_sample_data()
    
    # Display metrics cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.metric(
            label="1-Month Recession Probability",
            value=f"{df['recession_probability_1m'].iloc[-1]:.1%}",
            delta=f"{(df['recession_probability_1m'].iloc[-1] - df['recession_probability_1m'].iloc[-2]):.1%}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.metric(
            label="3-Month Recession Probability",
            value=f"{df['recession_probability_3m'].iloc[-1]:.1%}",
            delta=f"{(df['recession_probability_3m'].iloc[-1] - df['recession_probability_3m'].iloc[-2]):.1%}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.metric(
            label="6-Month Recession Probability",
            value=f"{df['recession_probability_6m'].iloc[-1]:.1%}",
            delta=f"{(df['recession_probability_6m'].iloc[-1] - df['recession_probability_6m'].iloc[-2]):.1%}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Plot recession probabilities
    st.markdown("<h2 class='sub-header'>Recession Probability Trends</h2>", unsafe_allow_html=True)
    
    fig, ax = plt.figure(figsize=(12, 6)), plt.subplot()
    ax.plot(df.index, df['recession_probability_1m'], label='1-Month', linewidth=2, color='#1E88E5')
    ax.plot(df.index, df['recession_probability_3m'], label='3-Month', linewidth=2, color='#FFA000')
    ax.plot(df.index, df['recession_probability_6m'], label='6-Month', linewidth=2, color='#D81B60')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add horizontal reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("<p class='info-text'>The chart shows the recession probability trends over time. A probability above 50% indicates a higher likelihood of a recession occurring within the specified time frame.</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display key economic indicators
    st.markdown("<h2 class='sub-header'>Key Economic Indicators</h2>", unsafe_allow_html=True)
    
    # Create 3 columns for indicators
    col1, col2, col3 = st.columns(3)
    
    indicators_list = list(indicators.items())
    
    with col1:
        for name, value in indicators_list[:4]:
            st.markdown(f"**{name}:** {value}")
    
    with col2:
        for name, value in indicators_list[4:8]:
            st.markdown(f"**{name}:** {value}")
    
    with col3:
        for name, value in indicators_list[8:]:
            st.markdown(f"**{name}:** {value}")
    
    # Yield Curve visualization
    st.markdown("---")
    st.markdown("<h2 class='sub-header'>Treasury Yield Curve</h2>", unsafe_allow_html=True)
    
    # Create yield curve visualization
    fig, ax = plt.figure(figsize=(10, 5)), plt.subplot()
    
    maturities = ['3-Month', '6-Month', '1-Year', '10-Year']
    rates = [indicators['3-Month Rate'], indicators['6-Month Rate'], 
             indicators['1-Year Rate'], indicators['10-Year Rate']]
    
    ax.plot(maturities, rates, marker='o', linewidth=2, color='#1E88E5')
    ax.set_ylabel('Yield Rate (%)')
    ax.set_title('Treasury Yield Curve')
    ax.grid(True, alpha=0.3)
    
    # Annotate the yield curve inversion if it exists
    if indicators['3-Month Rate'] > indicators['10-Year Rate']:
        ax.annotate('Yield Curve Inversion\n(Recession Signal)', 
                   xy=(3, rates[3]), 
                   xytext=(2.5, rates[3] - 0.5),
                   arrowprops=dict(facecolor='red', shrink=0.05),
                   color='red')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("<p class='info-text'>The yield curve shows the relationship between treasury bond maturities and their yields. An inverted yield curve (when short-term rates are higher than long-term rates) is often considered a recession indicator.</p>", unsafe_allow_html=True)

def display_custom_prediction():
    st.markdown("<h1 class='main-header'>Custom Recession Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Adjust the economic indicators below to generate a custom recession probability forecast.</p>", unsafe_allow_html=True)
    
    # Load sample data
    _, default_indicators = load_sample_data()
    
    # Create form for user input
    with st.form("custom_prediction_form"):
        st.markdown("<h2 class='sub-header'>Economic Indicators</h2>", unsafe_allow_html=True)
        
        # Create 3 columns with indicators
        col1, col2, col3 = st.columns(3)
        
        # Dictionary to store user inputs
        user_inputs = {}
        
        with col1:
            user_inputs['1-Year Rate'] = st.slider(
                "1-Year Treasury Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(default_indicators['1-Year Rate']),
                step=0.01
            )
            
            user_inputs['3-Month Rate'] = st.slider(
                "3-Month Treasury Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(default_indicators['3-Month Rate']),
                step=0.01
            )
            
            user_inputs['6-Month Rate'] = st.slider(
                "6-Month Treasury Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(default_indicators['6-Month Rate']),
                step=0.01
            )
            
            user_inputs['10-Year Rate'] = st.slider(
                "10-Year Treasury Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(default_indicators['10-Year Rate']),
                step=0.01
            )
        
        with col2:
            user_inputs['CPI'] = st.slider(
                "Consumer Price Index (% change)",
                min_value=-2.0,
                max_value=10.0,
                value=float(default_indicators['CPI']),
                step=0.1
            )
            
            user_inputs['PPI'] = st.slider(
                "Producer Price Index (% change)",
                min_value=-2.0,
                max_value=10.0,
                value=float(default_indicators['PPI']),
                step=0.1
            )
            
            user_inputs['Unemployment Rate'] = st.slider(
                "Unemployment Rate (%)",
                min_value=2.0,
                max_value=15.0,
                value=float(default_indicators['Unemployment Rate']),
                step=0.1
            )
            
            user_inputs['Industrial Production'] = st.slider(
                "Industrial Production Index",
                min_value=80.0,
                max_value=120.0,
                value=float(default_indicators['Industrial Production']),
                step=0.1
            )
        
        with col3:
            user_inputs['Share Price'] = st.slider(
                "Share Price Index",
                min_value=3000.0,
                max_value=5500.0,
                value=float(default_indicators['Share Price']),
                step=1.0
            )
            
            user_inputs['OECD CLI Index'] = st.slider(
                "OECD CLI Index",
                min_value=95.0,
                max_value=105.0,
                value=float(default_indicators['OECD CLI Index']),
                step=0.1
            )
            
            user_inputs['CSI Index'] = st.slider(
                "Consumer Sentiment Index",
                min_value=50.0,
                max_value=110.0,
                value=float(default_indicators['CSI Index']),
                step=0.1
            )
        
        # Submit button
        submitted = st.form_submit_button("Generate Prediction")
    
    # Process prediction if form is submitted
    if submitted:
        st.markdown("---")
        st.markdown("<h2 class='sub-header'>Recession Probability Forecast</h2>", unsafe_allow_html=True)
        
        # Get prediction
        predictions = predict_custom_recession(user_inputs)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
            prob_1m = predictions['1-Month']
            st.metric(
                label="1-Month Recession Probability",
                value=f"{prob_1m:.1%}"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
            prob_3m = predictions['3-Month']
            st.metric(
                label="3-Month Recession Probability",
                value=f"{prob_3m:.1%}"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
            prob_6m = predictions['6-Month']
            st.metric(
                label="6-Month Recession Probability",
                value=f"{prob_6m:.1%}"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualization
        st.markdown("---")
        
        # Create bar chart
        fig, ax = plt.figure(figsize=(10, 6)), plt.subplot()
        
        horizons = ['1-Month', '3-Month', '6-Month']
        probs = [predictions[h] for h in horizons]
        
        # Add color based on probability
        bar_colors = []
        for p in probs:
            if p < 0.3:
                bar_colors.append('#4CAF50')  # Green
            elif p < 0.7:
                bar_colors.append('#FFC107')  # Yellow/Amber
            else:
                bar_colors.append('#F44336')  # Red
        
        bars = ax.bar(horizons, probs, color=bar_colors)
        
        # Add probability values on top of bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.1%}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Probability')
        ax.set_title('Recession Probability by Time Horizon')
        
        # Add horizontal reference lines
        ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretation
        st.markdown("<h3 class='sub-header'>Interpretation</h3>", unsafe_allow_html=True)
        
        # Overall risk assessment
        avg_prob = (prob_1m + prob_3m + prob_6m) / 3
        if avg_prob < 0.3:
            risk_level = "Low"
            risk_class = "success"
        elif avg_prob < 0.7:
            risk_level = "Moderate"
            risk_class = "warning"
        else:
            risk_level = "High"
            risk_class = "warning"
        
        st.markdown(f"<p class='info-text'>Based on the provided economic indicators, the overall recession risk is <span class='{risk_class}'>{risk_level}</span>.</p>", unsafe_allow_html=True)
        
        # Detailed interpretation
        st.markdown("<p class='info-text'>Key factors influencing this prediction:</p>", unsafe_allow_html=True)
        
        # List factors with the most impact (simplified for demo)
        factors = []
        
        # Yield curve inversion check
        if user_inputs['3-Month Rate'] > user_inputs['10-Year Rate']:
            factors.append("Inverted yield curve (3-Month > 10-Year rate)")
        
        if user_inputs['Unemployment Rate'] > 5.0:
            factors.append("Elevated unemployment rate")
        
        if user_inputs['CPI'] > 4.0:
            factors.append("High inflation (CPI)")
        
        if user_inputs['PPI'] > 4.0:
            factors.append("High producer inflation (PPI)")
        
        if user_inputs['CSI Index'] < 70.0:
            factors.append("Low consumer sentiment")
        
        if user_inputs['OECD CLI Index'] < 99.0:
            factors.append("Below-trend OECD CLI (leading indicator)")
        
        if user_inputs['Industrial Production'] < 100.0:
            factors.append("Below-average industrial production")
        
        if user_inputs['Share Price'] < 4000.0:
            factors.append("Depressed share prices")
        
        if not factors:
            factors = ["Balanced economic conditions"]
        
        for factor in factors:
            st.markdown(f"- {factor}")
        
        # Yield Curve visualization
        st.markdown("---")
        st.markdown("<h3 class='sub-header'>Treasury Yield Curve Analysis</h3>", unsafe_allow_html=True)
        
        # Create yield curve visualization for user inputs
        fig, ax = plt.figure(figsize=(10, 5)), plt.subplot()
        
        maturities = ['3-Month', '6-Month', '1-Year', '10-Year']
        rates = [user_inputs['3-Month Rate'], user_inputs['6-Month Rate'], 
                 user_inputs['1-Year Rate'], user_inputs['10-Year Rate']]
        
        ax.plot(maturities, rates, marker='o', linewidth=2, color='#1E88E5')
        ax.set_ylabel('Yield Rate (%)')
        ax.set_title('Custom Treasury Yield Curve')
        ax.grid(True, alpha=0.3)
        
        # Annotate the yield curve inversion if it exists
        if user_inputs['3-Month Rate'] > user_inputs['10-Year Rate']:
            ax.annotate('Yield Curve Inversion\n(Recession Signal)', 
                       xy=(3, rates[3]), 
                       xytext=(2.5, rates[3] - 0.5),
                       arrowprops=dict(facecolor='red', shrink=0.05),
                       color='red')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        if user_inputs['3-Month Rate'] > user_inputs['10-Year Rate']:
            st.markdown("<p class='info-text warning'>Your inputs show an inverted yield curve, which has historically been a strong predictor of recessions within the next 6-18 months.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='info-text'>Your inputs show a normal yield curve, which typically indicates economic expansion.</p>", unsafe_allow_html=True)

def display_information():
    st.markdown("<h1 class='main-header'>Information</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>About Recession Radar</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>
    Recession Radar is a dashboard for visualizing and forecasting the probability of a recession occurring within different time horizons based on economic indicators.
    </p>
    
    <p class='info-text'>
    This simplified version uses synthetic data to demonstrate how the dashboard would work with real data and trained models.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>
    In the full version, Recession Radar uses machine learning models trained on historical economic data to predict the probability of a recession occurring within 1, 3, and 6 months.
    </p>
    
    <p class='info-text'>
    The models take into account various economic indicators, including:
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - Treasury Yields (1-Year, 3-Month, 6-Month, 10-Year)
    - Consumer Price Index (CPI)
    - Producer Price Index (PPI)
    - Industrial Production
    - Share Prices
    - Unemployment Rate
    - OECD Composite Leading Indicator
    - Consumer Sentiment Index
    """)
    
    st.markdown("<h2 class='sub-header'>Key Indicators Explained</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    - **Yield Curve Inversion**: When short-term interest rates (e.g., 3-Month) exceed long-term rates (e.g., 10-Year), it's historically been a reliable recession predictor
    
    - **OECD CLI Index**: The Composite Leading Indicator is designed to provide early signals of turning points in business cycles
    
    - **CSI Index**: Consumer Sentiment Index measures consumer confidence about the economy
    
    - **CPI and PPI**: Measure inflation from consumer and producer perspectives
    """)
    
    st.markdown("<h2 class='sub-header'>Interpretation</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>
    The recession probabilities should be interpreted as follows:
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - **Less than 30%**: Low recession risk
    - **30% to 70%**: Moderate recession risk
    - **Above 70%**: High recession risk
    """)
    
    st.markdown("<h2 class='sub-header'>Disclaimer</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>
    This application provides recession probability forecasts based on economic indicators. These forecasts are not financial advice and should not be used as the sole basis for financial decisions.
    </p>
    
    <p class='info-text'>
    The predictions are based on historical patterns and current economic conditions, but cannot account for unforeseen events or structural changes in the economy.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Data Sources</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>
    In the full version, the economic data used for training the models and generating forecasts is obtained from the following sources:
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - Federal Reserve Economic Data (FRED)
    - U.S. Bureau of Economic Analysis
    - U.S. Bureau of Labor Statistics
    - Organisation for Economic Co-operation and Development (OECD)
    - University of Michigan Consumer Sentiment Index
    - Standard & Poor's
    """)

if __name__ == "__main__":
    main()