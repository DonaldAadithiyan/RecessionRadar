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

# Apply custom CSS for dark theme
st.markdown("""
<style>
    /* Overall theme */
    .stApp {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        margin-bottom: 0.5rem !important;
    }
    .sub-header {
        font-size: 1.4rem !important;
        font-weight: 500 !important;
        color: #e0e0e0 !important;
        margin-bottom: 0.5rem !important;
    }
    .info-text {
        font-size: 0.9rem !important;
        color: #b0b0b0 !important;
    }
    
    /* Cards and containers */
    .highlight {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 0.4rem;
        border: 1px solid #3d3d3d;
    }
    .metrics-card {
        background-color: #2d2d2d;
        border-radius: 0.4rem;
        padding: 1.2rem;
        border: 1px solid #3d3d3d;
        box-shadow: 0 0.1rem 0.3rem rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    .metrics-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 0.2rem 0.4rem rgba(0, 0, 0, 0.4);
    }
    
    /* Status colors */
    .warning {
        color: #ff6b6b;
        font-weight: 600;
    }
    .success {
        color: #4ade80;
        font-weight: 600;
    }
    .info {
        color: #38bdf8;
        font-weight: 600;
    }
    
    /* Override Streamlit defaults */
    .stTextInput > div > div > input {
        background-color: #333333;
        color: #f0f0f0;
    }
    .stSelectbox > div > div > select {
        background-color: #333333;
        color: #f0f0f0;
    }
    .stNumberInput > div > div > input {
        background-color: #333333;
        color: #f0f0f0;
    }
    
    /* Chart background */
    .stPlotlyChart {
        background-color: #2d2d2d !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #252525;
    }
    
    /* Adjust metric styles for dark theme */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    div[data-testid="stMetricDelta"] svg {
        color: #4ade80 !important;
    }
    div[data-testid="stMetricDeltaIcon"] {
        color: #4ade80 !important;
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
    # Set dark background style
    plt.style.use('dark_background')
    fig, ax = plt.figure(figsize=(10, 6)), plt.subplot()
    
    # Dark theme colors
    background_color = '#1e1e1e'
    grid_color = '#333333'
    text_color = '#e0e0e0'
    line_color = '#FFA000'  # Orange color like in screenshot
    inversion_color = '#ff5252'
    
    # Set figure background
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
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
    
    # Plot the yield curve with professional styling
    ax.plot(maturities, rates, marker='o', linewidth=3, color=line_color)
    
    # Customize grid and spines
    ax.grid(True, alpha=0.2, color=grid_color, linestyle='--')
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    
    # Style labels and title
    ax.set_ylabel('Yield (%)', color=text_color, fontsize=10)
    ax.set_xlabel('Maturity', color=text_color, fontsize=10)
    ax.set_title(title, color=text_color, fontsize=14, pad=10)
    
    # Style tick labels
    ax.tick_params(colors=text_color)
    
    # Check for inversions
    has_inversion = False
    if highlight_inversion and len(maturities) >= 2:
        # Look for any case where a shorter-term rate is higher than a longer-term rate
        inversions = []
        for i in range(len(maturities) - 1):
            for j in range(i + 1, len(maturities)):
                if rates[i] > rates[j]:
                    inversions.append((i, j))
                    has_inversion = True
        
        # If there's an inversion, highlight the most significant one
        if inversions:
            # Find the inversion with the largest gap between maturity positions
            most_significant = max(inversions, key=lambda x: x[1] - x[0])
            i, j = most_significant
            
            # Add shading to visualize the inversion - more subtle than before
            ax.fill_between(maturities[i:j+1], rates[i:j+1], alpha=0.15, color=inversion_color)
    
    # Add inversion indicator in top-right corner if inversion exists
    if has_inversion:
        ax.text(0.97, 0.97, "Inverted Yield Curve Detected", 
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(facecolor='#541414', alpha=0.8, boxstyle='round,pad=0.5'),
                color=inversion_color)
    
    # Use tighter layout to maximize plot area
    plt.tight_layout()
    return fig

# Main app
def main():
    # Create sidebar with dark theme
    with st.sidebar:
        # Add custom CSS to style sidebar
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #252525;
            }
            .sidebar-title {
                font-size: 1.5rem !important;
                font-weight: 600 !important;
                color: #ffffff !important;
                margin-bottom: 1rem !important;
            }
            .sidebar-item {
                padding: 0.5rem 0;
                border-bottom: 1px solid #3d3d3d;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Header with logo
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="https://img.icons8.com/fluency/96/000000/economic-improvement.png" width="40">
            <h1 class="sidebar-title" style="margin-left: 10px; margin-bottom: 0 !important;">RecessionRadar</h1>
        </div>
        <div style="font-size: 0.8rem; color: #b0b0b0; margin-bottom: 1.5rem;">Economic Analysis Tool</div>
        """, unsafe_allow_html=True)
        
        # Navigation - styled to look like the screenshot
        st.markdown("<div style='margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
        page = st.radio("", ["Dashboard", "Custom Prediction", "Information"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add a "Rerun Pipeline" button with better styling
        st.markdown("<div class='sidebar-item'>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", help="Refresh all data from FRED and recalculate indicators"):
            # Clear the cache for the fetch_treasury_yields function
            fetch_treasury_yields.clear()
            st.success("✅ Data refreshed successfully")
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Check data source for yield curve
        st.markdown("<div class='sidebar-item'>", unsafe_allow_html=True)
        _, _, using_real_data = load_data()
        if using_real_data:
            st.markdown("<p style='color: #4ade80; font-size: 0.8rem;'>✓ Using real-time Treasury data</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #ff6b6b; font-size: 0.8rem;'>✗ Unable to fetch real-time data</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add some space
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        
        # Disclaimer in smaller font
        st.markdown("<div class='sidebar-item' style='font-size: 0.7rem; color: #888888;'>", unsafe_allow_html=True)
        st.markdown("**Disclaimer:** This application provides recession probability forecasts based on economic indicators. These forecasts are not financial advice.", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Version at bottom
        st.markdown("<div style='position: absolute; bottom: 10px; left: 10px; right: 10px; font-size: 0.7rem; color: #666666; text-align: center;'>", unsafe_allow_html=True)
        st.markdown("Version 1.0.0", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Page content
    if page == "Dashboard":
        display_dashboard()
    elif page == "Custom Prediction":
        display_custom_prediction()
    else:  # Information
        display_information()

def display_dashboard():
    # Main header with professional styling
    st.markdown("<h1 class='main-header'>RecessionRadar Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #b0b0b0; margin-bottom: 20px;'>Monitoring economic indicators and recession probabilities</p>", unsafe_allow_html=True)
    
    # Add info box for demo notice
    st.markdown("""
    <div style="display: flex; align-items: center; background-color: rgba(56, 189, 248, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 20px; border-left: 3px solid #38bdf8;">
        <div style="margin-right: 10px; color: #38bdf8;"><i>ℹ️</i></div>
        <div style="color: #b0b0b0; font-size: 0.9rem;">Currently showing hardcoded sample data. This will be replaced with real API data when connected to the backend.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df, indicators, using_real_data = load_data()
    
    # Display header for forecast section
    st.markdown("<h2 class='sub-header'>Recession Probability Forecast</h2>", unsafe_allow_html=True)
    
    # Display metrics cards in a more professional dark theme style
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metrics-card'>
            <h3 style='margin-top:0; font-size: 1.1rem; color: #e0e0e0;'>1-Month Forecast</h3>
            <p style='font-size: 2.2rem; font-weight: bold; margin: 0.2rem 0; color: #ff5252;'>
                {:.1%}
            </p>
            <p style='font-size: 0.8rem; color: #b0b0b0; margin: 0;'>
                Probability within next month
            </p>
        </div>
        """.format(df['recession_probability_1m'].iloc[-1]), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metrics-card'>
            <h3 style='margin-top:0; font-size: 1.1rem; color: #e0e0e0;'>3-Month Forecast</h3>
            <p style='font-size: 2.2rem; font-weight: bold; margin: 0.2rem 0; color: #ff5252;'>
                {:.1%}
            </p>
            <p style='font-size: 0.8rem; color: #b0b0b0; margin: 0;'>
                Probability within next three months
            </p>
        </div>
        """.format(df['recession_probability_3m'].iloc[-1]), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metrics-card'>
            <h3 style='margin-top:0; font-size: 1.1rem; color: #e0e0e0;'>6-Month Forecast</h3>
            <p style='font-size: 2.2rem; font-weight: bold; margin: 0.2rem 0; color: #ff5252;'>
                {:.1%}
            </p>
            <p style='font-size: 0.8rem; color: #b0b0b0; margin: 0;'>
                Probability within next six months
            </p>
        </div>
        """.format(df['recession_probability_6m'].iloc[-1]), unsafe_allow_html=True)
        
    # Add last updated timestamp
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S %p")
    st.caption(f"Last updated: {current_time}")
    
    st.markdown("---")
    
    # Plot recession probabilities
    st.markdown("<h2 class='sub-header'>Historical Recession Probability</h2>", unsafe_allow_html=True)
    
    # Set dark background style
    plt.style.use('dark_background')
    fig, ax = plt.figure(figsize=(12, 6)), plt.subplot()
    
    # Dark theme colors
    background_color = '#1e1e1e'
    grid_color = '#333333'
    text_color = '#e0e0e0'
    
    # Set figure background
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
    # Plot lines with more professional styling
    ax.plot(df.index, df['recession_probability_1m'], label='1-Month Probability', linewidth=2, color='#3a86ff')
    ax.plot(df.index, df['recession_probability_3m'], label='3-Month Probability', linewidth=2, color='#ffa000')
    ax.plot(df.index, df['recession_probability_6m'], label='6-Month Probability', linewidth=2, color='#ff5252')
    
    # Set y-axis range and add percentage markers
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '50%', '60%', '80%', '100%'])
    
    # Style grid and spines
    ax.grid(True, alpha=0.2, color=grid_color, linestyle='--')
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    
    # Add labels
    ax.set_ylabel('Probability (%)', color=text_color, fontsize=10)
    ax.set_xlabel('Date', color=text_color, fontsize=10)
    
    # Style tick labels
    ax.tick_params(colors=text_color)
    
    # Add horizontal reference lines
    ax.axhline(y=0.5, color='#888888', linestyle='--', alpha=0.5)
    ax.axhline(y=0.7, color='#ff5252', linestyle='--', alpha=0.5)
    
    # Add a better legend
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              frameon=True, ncol=3, facecolor=background_color, 
              edgecolor=grid_color, fontsize=10)
    for text in legend.get_texts():
        text.set_color(text_color)
    
    # Add title above chart
    ax.text(0.5, 1.05, "Recession Probability Over Time", 
            transform=ax.transAxes, fontsize=14, ha='center', 
            va='bottom', color=text_color)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add caption below chart
    st.caption("The chart shows the recession probability trends over time. A probability above 50% indicates a higher likelihood of a recession occurring within the specified time frame.")
    
    st.markdown("<p class='info-text'>The chart shows the recession probability trends over time. A probability above 50% indicates a higher likelihood of a recession occurring within the specified time frame.</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display key economic indicators with better styling
    st.markdown("<h2 class='sub-header'>Key Economic Indicators</h2>", unsafe_allow_html=True)
    
    # Create a stylish container for indicators
    st.markdown("""
    <style>
    .indicator-container {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        border-radius: 0.4rem;
        padding: 1rem;
    }
    .indicator-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #3d3d3d;
    }
    .indicator-item:last-child {
        border-bottom: none;
    }
    .indicator-name {
        color: #b0b0b0;
        font-size: 0.9rem;
    }
    .indicator-value {
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Filter out treasury rates that will be shown in the yield curve
    non_treasury_indicators = {k: v for k, v in indicators.items() if 'Rate' not in k}
    
    # Create columns for indicators
    col1, col2 = st.columns(2)
    
    # Prepare indicator lists for each column
    indicators_list = list(non_treasury_indicators.items())
    mid_point = len(indicators_list) // 2
    
    # Create first column of indicators
    with col1:
        st.markdown("<div class='indicator-container'>", unsafe_allow_html=True)
        for name, value in indicators_list[:mid_point]:
            # Format value with appropriate styling
            if name == 'CPI' or name == 'PPI':
                formatted_value = f"{value}%"
            elif name == 'Unemployment Rate':
                formatted_value = f"{value}%"
            else:
                formatted_value = f"{value}"
                
            st.markdown(f"""
            <div class="indicator-item">
                <div class="indicator-name">{name}</div>
                <div class="indicator-value">{formatted_value}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create second column of indicators
    with col2:
        st.markdown("<div class='indicator-container'>", unsafe_allow_html=True)
        for name, value in indicators_list[mid_point:]:
            # Format value with appropriate styling
            if name == 'CPI' or name == 'PPI':
                formatted_value = f"{value}%"
            elif name == 'Unemployment Rate':
                formatted_value = f"{value}%"
            else:
                formatted_value = f"{value}"
                
            st.markdown(f"""
            <div class="indicator-item">
                <div class="indicator-name">{name}</div>
                <div class="indicator-value">{formatted_value}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
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
    
    # Define the specific treasury rates we need based on our dataset columns
    required_rates = {
        '1-Year Rate': 4.86,  # Default values if real-time data isn't available
        '3-Month Rate': 4.52,
        '6-Month Rate': 4.65,
        '10-Year Rate': 3.81
    }
    
    # Update with any real values we have from FRED
    treasury_rates = {k: v for k, v in default_indicators.items() if 'Rate' in k}
    for rate in required_rates:
        if rate in treasury_rates:
            required_rates[rate] = treasury_rates[rate]
    
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
            # Include the specific rates we need for our model
            for rate_name, default_value in required_rates.items():
                user_inputs[rate_name] = get_numeric_input(
                    f"{rate_name} (%)",
                    default_value,
                    rate_name.replace('-', '').replace(' ', '_').lower()
                )
            
            # Debug info to show which rates are available from real-time data
            if st.checkbox("Show debug info"):
                st.write("Available rates from FRED:", list(treasury_rates.keys()))
        
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