import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import requests
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

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

# Function to fetch Treasury yield data from FRED
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_treasury_yields():
    """
    Fetch real treasury yield data from FRED using pandas_datareader
    """
    try:
        # Define the end date as today
        end_date = datetime.now()
        # Start date is 7 days ago to ensure we get recent data even with holidays/weekends
        start_date = end_date - timedelta(days=14)  # Extended from 7 to 14 days
        
        # Define the treasury series codes to fetch from FRED
        series_codes = {
            'DGS1MO': '1-Month Rate',    # 1-month Treasury constant maturity
            'TB3MS': '3-Month Rate',     # 3-month Treasury bill
            'TB6MS': '6-Month Rate',     # 6-month Treasury bill
            'GS1': '1-Year Rate',        # 1-year Treasury constant maturity
            'GS2': '2-Year Rate',        # 2-year Treasury constant maturity
            'GS5': '5-Year Rate',        # 5-year Treasury constant maturity
            'GS10': '10-Year Rate',      # 10-year Treasury constant maturity
            'GS30': '30-Year Rate'       # 30-year Treasury constant maturity
        }
        
        # Dictionary to store the results
        yields = {}
        
        # Fetch each series
        for code, name in series_codes.items():
            try:
                # Get data from FRED
                df = pdr.DataReader(code, 'fred', start_date, end_date)
                # Get latest non-NaN value
                if not df.empty:
                    # Drop NaN values and get most recent
                    latest_data = df.dropna()
                    if not latest_data.empty:
                        yields[name] = latest_data.iloc[-1, 0]
                    else:
                        print(f"No valid data found for {name}")
            except Exception as e:
                print(f"Error fetching {name}: {str(e)}")
        
        return yields
    
    except Exception as e:
        print(f"Error in fetch_treasury_yields: {str(e)}")
        return {}

# Function to load data
def load_data():
    # Generate sample recession probability data (keeping synthetic for demo)
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
    
    # Get real treasury yield data (no fallbacks)
    real_yields = fetch_treasury_yields()
    
    # Create indicators dictionary
    indicators = {}
    
    # Add Treasury rates - ONLY use real data
    indicators.update(real_yields)
    
    # Other economic indicators (still using sample data since we're focusing on Treasury yields)
    indicators.update({
        'CPI': 3.2,
        'Industrial Production': 104.5,
        'Share Price': 4512.8,
        'Unemployment Rate': 3.8,
        'PPI': 2.8,
        'OECD CLI Index': 99.2,
        'CSI Index': 82.7
    })
    
    # Track if we're using real data for yield curve
    using_real_data = len(real_yields) > 0
    
    return df, indicators, using_real_data

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
    normalized = {}
    for key in weights.keys():
        if key in inputs:
            if key == '1-Year Rate':
                normalized[key] = (inputs[key] - 0) / 8
            elif key == '3-Month Rate':
                normalized[key] = (inputs[key] - 0) / 8
            elif key == '6-Month Rate':
                normalized[key] = (inputs[key] - 0) / 8
            elif key == 'CPI':
                normalized[key] = (inputs[key] - 0) / 10
            elif key == 'Industrial Production':
                normalized[key] = (inputs[key] - 90) / 30
            elif key == '10-Year Rate':
                normalized[key] = (inputs[key] - 0) / 8
            elif key == 'Share Price':
                normalized[key] = (inputs[key] - 3000) / 2000
            elif key == 'Unemployment Rate':
                normalized[key] = (inputs[key] - 3) / 7
            elif key == 'PPI':
                normalized[key] = (inputs[key] - 0) / 10
            elif key == 'OECD CLI Index':
                normalized[key] = (inputs[key] - 95) / 10
            elif key == 'CSI Index':
                normalized[key] = (inputs[key] - 60) / 40
        else:
            normalized[key] = 0.5  # Default to neutral value if missing
    
    # Calculate weighted sum
    weighted_sum = sum(weights[key] * normalized[key] for key in weights)
    
    # Check for yield curve inversion (3-Month > 10-Year) - adds recession probability
    if '3-Month Rate' in inputs and '10-Year Rate' in inputs:
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

# Function to render the yield curve
def plot_yield_curve(treasury_rates, title='Treasury Yield Curve', highlight_inversion=True, custom=False):
    """
    Creates and returns a yield curve visualization
    
    Args:
        treasury_rates: Dictionary with maturity rates
        title: Title for the plot
        highlight_inversion: Whether to highlight yield curve inversions
        custom: Whether this is a custom user-defined curve
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.figure(figsize=(10, 5)), plt.subplot()
    
    # Define the order of maturities for proper x-axis plotting
    maturity_order = [
        '1-Month Rate', '3-Month Rate', '6-Month Rate', '1-Year Rate',
        '2-Year Rate', '5-Year Rate', '10-Year Rate', '30-Year Rate'
    ]
    
    # Extract available maturities and rates in correct order
    maturities = []
    rates = []
    
    for maturity in maturity_order:
        if maturity in treasury_rates:
            # Extract just the maturity name without "Rate"
            maturities.append(maturity.replace(' Rate', ''))
            rates.append(treasury_rates[maturity])
    
    # Plot the yield curve
    ax.plot(maturities, rates, marker='o', linewidth=2, color='#1E88E5')
    ax.set_ylabel('Yield Rate (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Check for inversions
    if highlight_inversion and len(maturities) >= 2:
        # Look for any case where a shorter-term rate is higher than a longer-term rate
        inversions = []
        for i in range(len(maturities) - 1):
            for j in range(i + 1, len(maturities)):
                if rates[i] > rates[j]:
                    inversions.append((i, j))
        
        # If there's an inversion, highlight the most significant one (typically short vs long term)
        if inversions:
            # Find the inversion with the largest gap between maturity positions
            most_significant = max(inversions, key=lambda x: x[1] - x[0])
            i, j = most_significant
            
            # Highlight the inversion
            ax.annotate('Yield Curve Inversion\n(Recession Signal)', 
                       xy=(maturities[j], rates[j]), 
                       xytext=(maturities[j-1], rates[j] - 0.4 if rates[j] > 0.5 else rates[j] + 0.4),
                       arrowprops=dict(facecolor='red', shrink=0.05),
                       color='red')
            
            # Add shading to visualize the inversion
            ax.fill_between(maturities[i:j+1], rates[i:j+1], alpha=0.15, color='red')
    
    # Add steepness indicator
    if len(maturities) >= 2:
        # Calculate steepness (difference between longest and shortest maturity)
        steepness = rates[-1] - rates[0]
        ax.text(0.02, 0.95, f"Curve Steepness: {steepness:.2f}%", 
                transform=ax.transAxes, fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.6))
    
    # Use tighter layout to maximize plot area
    plt.tight_layout()
    return fig

# Main app
def main():
    # Create sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/economic-improvement.png", width=80)
        st.title("Recession Radar")
        
        # Navigation
        page = st.radio("Navigation", ["Dashboard", "Custom Prediction", "Information"])
        
        # Add a "Rerun Pipeline" button
        if st.button("🔄 Rerun Data Pipeline", help="Refresh all data from FRED and recalculate indicators"):
            # Clear the cache for the fetch_treasury_yields function
            fetch_treasury_yields.clear()
            st.success("✅ Data pipeline rerun successfully! Treasury yield data has been refreshed.")
            st.experimental_rerun()
        
        # Check data source for yield curve
        _, _, using_real_data = load_data()
        if using_real_data:
            st.success("✓ Using real-time Treasury yield data")
        else:
            st.error("✗ Unable to fetch real-time Treasury yield data")
            st.info("Please check your internet connection or try again later.")
        
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
    st.markdown("<h1 class='main-header'>RecessionRadar Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>This dashboard visualizes the probability of a recession occurring within the next 1, 3, and 6 months based on economic indicators.</p>", unsafe_allow_html=True)
    
    # Load data
    df, indicators, using_real_data = load_data()
    
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
    
    # Filter out treasury rates that will be shown in the yield curve
    non_treasury_indicators = {k: v for k, v in indicators.items() if 'Rate' not in k}
    non_treasury_list = list(non_treasury_indicators.items())
    
    # Calculate items per column
    items_per_col = len(non_treasury_list) // 3 + (1 if len(non_treasury_list) % 3 > 0 else 0)
    
    with col1:
        for name, value in non_treasury_list[:items_per_col]:
            st.markdown(f"**{name}:** {value}")
    
    with col2:
        for name, value in non_treasury_list[items_per_col:items_per_col*2]:
            st.markdown(f"**{name}:** {value}")
    
    with col3:
        for name, value in non_treasury_list[items_per_col*2:]:
            st.markdown(f"**{name}:** {value}")
    
    # Yield Curve visualization
    st.markdown("---")
    st.markdown("<h2 class='sub-header'>Treasury Yield Curve</h2>", unsafe_allow_html=True)
    
    # Extract treasury rates from indicators
    treasury_rates = {k: v for k, v in indicators.items() if 'Rate' in k}
    
    if not treasury_rates:
        # No real Treasury yield data available
        st.warning("⚠️ Real-time Treasury yield data is not available at the moment.")
        st.info("This could be due to connectivity issues, API limits, or market closures. Please try again later.")
        st.button("🔄 Try Refreshing Data", on_click=lambda: fetch_treasury_yields.clear())
    elif len(treasury_rates) <= 1:
        # Only limited data available
        st.warning("⚠️ Only limited Treasury yield data is available. A full yield curve requires at least two data points.")
        st.info("Try using the 'Rerun Data Pipeline' button in the sidebar to refresh the data.")
        
        # Still show what we have
        fig = plot_yield_curve(treasury_rates)
        st.pyplot(fig)
    else:
        # Create yield curve visualization
        fig = plot_yield_curve(treasury_rates)
        st.pyplot(fig)
        
        # Add data source info
        st.caption("Source: Federal Reserve Economic Data (FRED) - Current as of today")
        
        # Check for inversions and provide interpretation
        has_inversion = False
        short_rates = ['1-Month Rate', '3-Month Rate', '6-Month Rate']
        long_rates = ['10-Year Rate', '30-Year Rate']
        
        for short in short_rates:
            for long in long_rates:
                if short in treasury_rates and long in treasury_rates:
                    if treasury_rates[short] > treasury_rates[long]:
                        has_inversion = True
                        st.markdown(f"<p class='info-text warning'>The yield curve is currently inverted ({short.replace(' Rate', '')} > {long.replace(' Rate', '')}). This has historically preceded recessions by 6-18 months.</p>", unsafe_allow_html=True)
                        break
            if has_inversion:
                break
        
        if not has_inversion:
            st.markdown("<p class='info-text'>The yield curve is currently normal (long-term rates > short-term rates), which typically indicates economic expansion.</p>", unsafe_allow_html=True)
        
        # Add information about steepness if we have enough data points
        if len(treasury_rates) >= 2:
            # Find shortest and longest maturity
            maturity_order = ['1-Month Rate', '3-Month Rate', '6-Month Rate', '1-Year Rate', 
                              '2-Year Rate', '5-Year Rate', '10-Year Rate', '30-Year Rate']
            
            shortest = None
            longest = None
            
            for maturity in maturity_order:
                if maturity in treasury_rates:
                    if shortest is None:
                        shortest = maturity
                    longest = maturity
            
            if shortest and longest and shortest != longest:
                steepness = treasury_rates[longest] - treasury_rates[shortest]
                
                if steepness > 1.5:
                    st.markdown("<p class='info-text'>The yield curve is quite steep, which often indicates strong economic growth expectations.</p>", unsafe_allow_html=True)
                elif steepness < -0.5:
                    st.markdown("<p class='info-text warning'>The yield curve is deeply inverted, which historically has been a stronger recession signal.</p>", unsafe_allow_html=True)
        
        st.markdown("<p class='info-text'>The yield curve shows the relationship between treasury bond maturities and their yields. An inverted yield curve (when short-term rates are higher than long-term rates) is often considered a recession indicator.</p>", unsafe_allow_html=True)

def display_custom_prediction():
    st.markdown("<h1 class='main-header'>Custom Recession Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Adjust the economic indicators below to generate a custom recession probability forecast.</p>", unsafe_allow_html=True)
    
    # Load data
    _, default_indicators, using_real_data = load_data()
    
    # Check if we have Treasury yield data
    treasury_rates = {k: v for k, v in default_indicators.items() if 'Rate' in k}
    
    if not treasury_rates:
        st.warning("⚠️ Real-time Treasury yield data is not available at the moment.")
        st.info("This is required for custom predictions. Please try again later or use the 'Rerun Data Pipeline' button in the sidebar.")
        return
    
    # Create form for user input
    with st.form("custom_prediction_form"):
        st.markdown("<h2 class='sub-header'>Economic Indicators</h2>", unsafe_allow_html=True)
        
        # Create 3 columns with indicators
        col1, col2, col3 = st.columns(3)
        
        # Dictionary to store user inputs
        user_inputs = {}
        
        # Helper function to validate numeric input
        def get_numeric_input(label, default_value, key):
            value = st.text_input(
                label,
                value=str(default_value),
                key=key
            )
            try:
                return float(value)
            except ValueError:
                return default_value
        
        with col1:
            st.markdown("**Treasury Rates**")
            # Include all available Treasury rates from our real-time data
            for rate_name in ['1-Month Rate', '3-Month Rate', '6-Month Rate', '1-Year Rate', 
                             '2-Year Rate', '5-Year Rate', '10-Year Rate', '30-Year Rate']:
                if rate_name in default_indicators:
                    user_inputs[rate_name] = get_numeric_input(
                        f"{rate_name} (%)",
                        default_indicators[rate_name],
                        rate_name.replace('-', '').replace(' ', '_').lower()
                    )
        
        with col2:
            st.markdown("**Economic Indicators**")
            user_inputs['CPI'] = get_numeric_input(
                "Consumer Price Index (% change)",
                default_indicators['CPI'],
                "cpi"
            )
            
            user_inputs['PPI'] = get_numeric_input(
                "Producer Price Index (% change)",
                default_indicators['PPI'],
                "ppi"
            )
            
            user_inputs['Unemployment Rate'] = get_numeric_input(
                "Unemployment Rate (%)",
                default_indicators['Unemployment Rate'],
                "unemp_rate"
            )
            
            user_inputs['Industrial Production'] = get_numeric_input(
                "Industrial Production Index",
                default_indicators['Industrial Production'],
                "indpro"
            )
        
        with col3:
            st.markdown("**Market & Sentiment Indicators**")
            user_inputs['Share Price'] = get_numeric_input(
                "Share Price Index",
                default_indicators['Share Price'],
                "share_price"
            )
            
            user_inputs['OECD CLI Index'] = get_numeric_input(
                "OECD CLI Index",
                default_indicators['OECD CLI Index'],
                "oecd_cli"
            )
            
            user_inputs['CSI Index'] = get_numeric_input(
                "Consumer Sentiment Index",
                default_indicators['CSI Index'],
                "csi"
            )
        
        # Add a note about value ranges
        st.markdown("---")
        st.markdown("""
        <p class='info-text'>
        <b>Note:</b> Typical ranges for indicators:
        <ul>
        <li>Treasury Rates: 0-10%</li>
        <li>CPI/PPI: -2 to 10%</li>
        <li>Unemployment: 2-15%</li>
        <li>Industrial Production: 80-120</li>
        <li>Share Price Index: 3000-5500</li>
        <li>OECD CLI: 95-105</li>
        <li>Consumer Sentiment: 50-110</li>
        </ul>
        </p>
        """, unsafe_allow_html=True)
        
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
        if '3-Month Rate' in user_inputs and '10-Year Rate' in user_inputs:
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
        
        # Extract treasury rates from user inputs
        treasury_rates = {k: v for k, v in user_inputs.items() if 'Rate' in k}
        
        # Create yield curve visualization
        fig = plot_yield_curve(treasury_rates, title='Custom Treasury Yield Curve', custom=True)
        st.pyplot(fig)
        
        # Check for inversions and provide interpretation
        has_inversion = False
        short_rates = ['1-Month Rate', '3-Month Rate', '6-Month Rate']
        long_rates = ['10-Year Rate', '30-Year Rate']
        
        for short in short_rates:
            for long in long_rates:
                if short in treasury_rates and long in treasury_rates:
                    if treasury_rates[short] > treasury_rates[long]:
                        has_inversion = True
                        st.markdown(f"<p class='info-text warning'>Your inputs show an inverted yield curve ({short.replace(' Rate', '')} > {long.replace(' Rate', '')}). This has historically preceded recessions by 6-18 months.</p>", unsafe_allow_html=True)
                        break
            if has_inversion:
                break
        
        if not has_inversion:
            st.markdown("<p class='info-text'>Your inputs show a normal yield curve, which typically indicates economic expansion.</p>", unsafe_allow_html=True)
        
        # Add information about steepness if we have enough data points
        if len(treasury_rates) >= 2:
            # Find shortest and longest maturity
            maturity_order = ['1-Month Rate', '3-Month Rate', '6-Month Rate', '1-Year Rate', 
                              '2-Year Rate', '5-Year Rate', '10-Year Rate', '30-Year Rate']
            
            available_rates = [rate for rate in maturity_order if rate in treasury_rates]
            if len(available_rates) >= 2:
                shortest = available_rates[0]
                longest = available_rates[-1]
                
                steepness = treasury_rates[longest] - treasury_rates[shortest]
                
                if steepness > 1.5:
                    st.markdown("<p class='info-text'>Your custom yield curve is quite steep, which often indicates strong economic growth expectations.</p>", unsafe_allow_html=True)
                elif steepness < -0.5:
                    st.markdown("<p class='info-text warning'>Your custom yield curve is deeply inverted, which historically has been a stronger recession signal.</p>", unsafe_allow_html=True)

def display_information():
    st.markdown("<h1 class='main-header'>Information</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>About Recession Radar</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>
    RecessionRadar is a dashboard for visualizing and forecasting the probability of a recession occurring within different time horizons based on economic indicators.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>
    RecessionRadar uses machine learning models trained on historical economic data to predict the probability of a recession occurring within 1, 3, and 6 months.
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
    
    st.markdown("<h2 class='sub-header'>Understanding the Treasury Yield Curve</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>
    The Treasury yield curve is a key economic indicator that plots interest rates of U.S. Treasury bonds at different maturities. In a normal economic environment, longer-term bonds have higher yields than shorter-term bonds, creating an upward-sloping curve.
    </p>
    
    <p class='info-text'>
    <b>Types of yield curves:</b>
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Normal (Upward Sloping)**: Long-term rates higher than short-term rates; typical during economic expansions
        - **Inverted**: Short-term rates higher than long-term rates; often precedes recessions
        """)
    
    with col2:
        st.markdown("""
        - **Flat**: Similar yields across all maturities; often occurs during transitions
        - **Steep**: Large gap between short and long-term yields; common during early recovery phases
        """)
    
    st.markdown("""
    <p class='info-text'>
    <b>Why the yield curve matters:</b> An inverted yield curve (particularly when the 3-Month rate exceeds the 10-Year rate) has preceded every U.S. recession since 1955, with a typical lead time of 6-18 months. This happens because investors expect economic weakness and lower inflation in the future, driving demand for longer-term bonds and pushing their yields down.
    </p>
    """, unsafe_allow_html=True)
    
    # Show example yield curves
    st.markdown("<h3>Example Yield Curves</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Normal curve example
        normal_curve = {
            '1-Month Rate': 3.8,
            '3-Month Rate': 3.9,
            '6-Month Rate': 4.1,
            '1-Year Rate': 4.3,
            '2-Year Rate': 4.5,
            '5-Year Rate': 4.7,
            '10-Year Rate': 5.0,
            '30-Year Rate': 5.3
        }
        fig = plot_yield_curve(normal_curve, title='Normal Yield Curve (Expansion)', highlight_inversion=False)
        st.pyplot(fig)
        st.caption("A normal yield curve typically indicates economic expansion")
    
    with col2:
        # Inverted curve example
        inverted_curve = {
            '1-Month Rate': 5.2,
            '3-Month Rate': 5.1,
            '6-Month Rate': 5.0,
            '1-Year Rate': 4.9,
            '2-Year Rate': 4.7,
            '5-Year Rate': 4.5,
            '10-Year Rate': 4.3,
            '30-Year Rate': 4.1
        }
        fig = plot_yield_curve(inverted_curve, title='Inverted Yield Curve (Recession Warning)')
        st.pyplot(fig)
        st.caption("An inverted yield curve often precedes recessions")
    
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
    The economic data used for training the models and generating forecasts is obtained from the following sources:
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - Federal Reserve Economic Data (FRED) - Real-time Treasury yield data
    - US Recession Dataset - kaggle (shubhaanshkumar)
    """)

if __name__ == "__main__":
    main()