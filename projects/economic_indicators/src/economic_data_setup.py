# ============================================
# JUPYTER NOTEBOOK SETUP FOR ECONOMIC FORECASTING PROJECT
# ============================================

# STEP 1: Install Required Packages
# Run this cell first (only need to run once)
!pip install pandas fredapi numpy matplotlib seaborn plotly prophet scikit-learn statsmodels

# ============================================
# STEP 2: Import Libraries
# Run this cell after installation
# ============================================

import pandas as pd
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")

# ============================================
# STEP 3: Set Up FRED API
# Get your free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
# ============================================

# Replace 'YOUR_API_KEY_HERE' with your actual FRED API key
FRED_API_KEY = 'YOUR_API_KEY_HERE'

# Initialize FRED API
fred = Fred(api_key=FRED_API_KEY)

print("✅ FRED API initialized!")

# ============================================
# STEP 4: Test the Connection
# ============================================

# Test by fetching unemployment rate data
try:
    unemployment = fred.get_series('UNRATE', observation_start='2020-01-01')
    print(f"✅ Successfully fetched {len(unemployment)} data points for unemployment rate")
    print(f"\nLatest unemployment rate: {unemployment.iloc[-1]:.1f}%")
    print(f"Date: {unemployment.index[-1].strftime('%B %Y')}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure you've replaced 'YOUR_API_KEY_HERE' with your actual API key")

# ============================================
# STEP 5: Fetch Multiple Economic Indicators
# ============================================

def fetch_economic_data(start_date='2010-01-01'):
    """
    Fetch key economic indicators from FRED
    
    Parameters:
    -----------
    start_date : str
        Start date for data collection (format: 'YYYY-MM-DD')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all economic indicators
    """
    
    indicators = {
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'GDP': 'Gross Domestic Product',
        'FEDFUNDS': 'Federal Funds Rate',
        'HOUST': 'Housing Starts',
        'UMCSENT': 'Consumer Sentiment',
        'INDPRO': 'Industrial Production',
        'PAYEMS': 'Total Nonfarm Payrolls'
    }
    
    data_dict = {}
    
    print("Fetching economic indicators from FRED...")
    for code, name in indicators.items():
        try:
            series = fred.get_series(code, observation_start=start_date)
            data_dict[name] = series
            print(f"✅ {name}: {len(series)} observations")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    # Combine into DataFrame
    df = pd.DataFrame(data_dict)
    
    print(f"\n✅ Total data shape: {df.shape}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return df

# Fetch the data
economic_data = fetch_economic_data()

# Display first few rows
print("\n📊 Sample of Economic Data:")
economic_data.head()

# ============================================
# STEP 6: Quick Data Visualization
# ============================================

def plot_economic_indicator(df, column_name):
    """
    Create a simple time series plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column_name], linewidth=2)
    plt.title(f'{column_name} Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(column_name, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example: Plot unemployment rate
if 'Unemployment Rate' in economic_data.columns:
    plot_economic_indicator(economic_data, 'Unemployment Rate')

# ============================================
# STEP 7: Data Summary Statistics
# ============================================

print("\n📈 Summary Statistics:")
print(economic_data.describe())

print("\n🔍 Missing Values:")
print(economic_data.isnull().sum())

print("\n📅 Data Frequency:")
print(f"Total observations: {len(economic_data)}")
print(f"Date range: {(economic_data.index.max() - economic_data.index.min()).days} days")

# ============================================
# BONUS: Save Data to CSV
# ============================================

# Save to CSV for future use
economic_data.to_csv('economic_indicators.csv')
print("\n💾 Data saved to 'economic_indicators.csv'")

print("\n" + "="*50)
print("🎉 SETUP COMPLETE!")
print("="*50)
print("\nNext steps:")
print("1. Explore the data further")
print("2. Create visualizations")
print("3. Build forecasting models")
print("4. Check out the project guide for detailed instructions!")