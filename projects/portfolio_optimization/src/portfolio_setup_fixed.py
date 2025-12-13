# ============================================
# S&P 500 PORTFOLIO OPTIMIZATION - FIXED SETUP
# Handles Wikipedia 403 error with multiple fallback methods
# ============================================

# ============================================
# STEP 1: INSTALL ALL REQUIRED PACKAGES
# ============================================

!pip install yfinance pandas numpy matplotlib seaborn plotly \
    scipy scikit-learn cvxpy PyPortfolioOpt streamlit \
    statsmodels quantstats tqdm jupyter requests beautifulsoup4 lxml html5lib

print("✅ All packages installed!")

# ============================================
# STEP 2: IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:.4f}'.format

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"yfinance version: {yf.__version__}")

# ============================================
# STEP 3: GET S&P 500 STOCK LIST (FIXED!)
# ============================================

def get_sp500_tickers_robust():
    """
    Get S&P 500 tickers with multiple fallback methods
    Handles 403 Forbidden errors
    """
    print("📥 Fetching S&P 500 stock list...")
    
    # METHOD 1: Wikipedia with proper headers (BEST)
    try:
        print("   Trying Method 1: Wikipedia with headers...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse with pandas
        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]
        
        print("   ✅ Method 1 successful!")
        
    except Exception as e:
        print(f"   ❌ Method 1 failed: {str(e)[:50]}...")
        
        # METHOD 2: Try with different parser
        try:
            print("   Trying Method 2: Alternative HTML parser...")
            response = requests.get(url, headers=headers, timeout=10)
            tables = pd.read_html(StringIO(response.text), flavor='html5lib')
            sp500_table = tables[0]
            print("   ✅ Method 2 successful!")
            
        except Exception as e2:
            print(f"   ❌ Method 2 failed: {str(e2)[:50]}...")
            
            # METHOD 3: Use yfinance S&P 500 ETF holdings
            try:
                print("   Trying Method 3: S&P 500 ETF (SPY) holdings...")
                spy = yf.Ticker("SPY")
                holdings = spy.get_institutional_holders()
                
                # This won't work perfectly, so fall to Method 4
                raise Exception("Moving to manual list")
                
            except:
                print("   ❌ Method 3 failed...")
                
                # METHOD 4: Manual curated list of major S&P 500 stocks
                print("   Using Method 4: Curated list of 100 major S&P 500 stocks...")
                
                sp500_table = pd.DataFrame({
                    'Symbol': [
                        # Technology (20)
                        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'TSLA', 
                        'ADBE', 'CRM', 'CSCO', 'ORCL', 'ACN', 'AMD', 'INTC', 'QCOM', 
                        'TXN', 'IBM', 'INTU', 'AMAT',
                        
                        # Financials (20)
                        'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'SPGI', 
                        'BLK', 'AXP', 'C', 'SCHW', 'CB', 'MMC', 'PGR', 'AON', 'CME',
                        'ICE', 'COF',
                        
                        # Healthcare (20)
                        'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 
                        'PFE', 'BMY', 'AMGN', 'GILD', 'CVS', 'CI', 'MDT', 'ISRG',
                        'REGN', 'VRTX', 'ZTS', 'ELV',
                        
                        # Consumer Discretionary (15)
                        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX',
                        'BKNG', 'MAR', 'GM', 'F', 'ABNB', 'CMG', 'ORLY',
                        
                        # Consumer Staples (10)
                        'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ',
                        'CL', 'KMB',
                        
                        # Energy (5)
                        'XOM', 'CVX', 'COP', 'EOG', 'SLB',
                        
                        # Industrials (10)
                        'UPS', 'RTX', 'HON', 'UNP', 'BA', 'CAT', 'GE', 'LMT',
                        'DE', 'MMM'
                    ],
                    'Security': [
                        # Technology
                        'Apple', 'Microsoft', 'NVIDIA', 'Alphabet Class A', 'Alphabet Class C',
                        'Meta', 'Broadcom', 'Tesla', 'Adobe', 'Salesforce', 'Cisco', 'Oracle',
                        'Accenture', 'AMD', 'Intel', 'Qualcomm', 'Texas Instruments', 'IBM',
                        'Intuit', 'Applied Materials',
                        
                        # Financials
                        'Berkshire Hathaway', 'JPMorgan', 'Visa', 'Mastercard', 'Bank of America',
                        'Wells Fargo', 'Morgan Stanley', 'Goldman Sachs', 'S&P Global', 'BlackRock',
                        'American Express', 'Citigroup', 'Charles Schwab', 'Chubb', 'Marsh McLennan',
                        'Progressive', 'Aon', 'CME Group', 'ICE', 'Capital One',
                        
                        # Healthcare
                        'UnitedHealth', 'Johnson & Johnson', 'Eli Lilly', 'AbbVie', 'Merck',
                        'Thermo Fisher', 'Abbott', 'Danaher', 'Pfizer', 'Bristol Myers',
                        'Amgen', 'Gilead', 'CVS Health', 'Cigna', 'Medtronic', 'Intuitive Surgical',
                        'Regeneron', 'Vertex', 'Zoetis', 'Elevance Health',
                        
                        # Consumer Discretionary
                        'Amazon', 'Tesla', 'Home Depot', 'McDonalds', 'Nike', 'Lowes', 'Starbucks',
                        'TJX Companies', 'Booking Holdings', 'Marriott', 'General Motors', 'Ford',
                        'Airbnb', 'Chipotle', 'O\'Reilly',
                        
                        # Consumer Staples
                        'Walmart', 'Procter & Gamble', 'Costco', 'Coca-Cola', 'PepsiCo',
                        'Philip Morris', 'Altria', 'Mondelez', 'Colgate-Palmolive', 'Kimberly-Clark',
                        
                        # Energy
                        'ExxonMobil', 'Chevron', 'ConocoPhillips', 'EOG Resources', 'Schlumberger',
                        
                        # Industrials
                        'UPS', 'Raytheon', 'Honeywell', 'Union Pacific', 'Boeing', 'Caterpillar',
                        'GE', 'Lockheed Martin', 'Deere', '3M'
                    ],
                    'GICS Sector': (
                        ['Technology'] * 20 + 
                        ['Financials'] * 20 + 
                        ['Healthcare'] * 20 +
                        ['Consumer Discretionary'] * 15 +
                        ['Consumer Staples'] * 10 +
                        ['Energy'] * 5 +
                        ['Industrials'] * 10
                    )
                })
                
                print("   ✅ Using curated list of 100 major stocks")
    
    # Clean ticker symbols
    sp500_table['Ticker'] = sp500_table['Symbol'].str.replace('.', '-', regex=False)
    
    print(f"\n✅ Successfully loaded {len(sp500_table)} stocks")
    print(f"\nSector Distribution:")
    print(sp500_table['GICS Sector'].value_counts())
    
    return sp500_table

# Get S&P 500 list
sp500_df = get_sp500_tickers_robust()

# Display sample
print("\n📊 Sample of Stocks:")
print(sp500_df[['Symbol', 'Security', 'GICS Sector']].head(10))

# Get tickers list
all_tickers = sp500_df['Ticker'].tolist()
print(f"\n📋 Total tickers available: {len(all_tickers)}")

# ============================================
# STEP 4: DOWNLOAD HISTORICAL DATA (ROBUST)
# ============================================

def download_stock_data_robust(tickers, start_date, end_date, sample_size=None):
    """
    Download historical price data with error handling
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : datetime
        Start date for data
    end_date : datetime
        End date for data
    sample_size : int, optional
        Number of random stocks to sample (None = all stocks)
    
    Returns:
    --------
    pd.DataFrame
        Adjusted close prices
    """
    # Sample stocks if requested
    if sample_size and sample_size < len(tickers):
        tickers = np.random.choice(tickers, sample_size, replace=False).tolist()
        print(f"\n📊 Using sample of {sample_size} stocks for faster processing")
    
    print(f"\n📥 Downloading data for {len(tickers)} stocks...")
    print(f"   Date range: {start_date.date()} to {end_date.date()}")
    print(f"   This may take 1-3 minutes...\n")
    
    # Method 1: Try bulk download (fastest)
    try:
        print("   Attempting bulk download...")
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=True,
            threads=True,
            group_by='ticker'
        )
        
        # Extract adjusted close prices
        if len(tickers) == 1:
            adj_close = data['Adj Close'].to_frame()
            adj_close.columns = tickers
        else:
            # Handle multi-column structure
            if isinstance(data.columns, pd.MultiIndex):
                adj_close = data.xs('Adj Close', axis=1, level=1)
            else:
                adj_close = data['Adj Close']
        
        print("\n   ✅ Bulk download successful!")
        
    except Exception as e:
        print(f"\n   ⚠️ Bulk download failed: {str(e)[:100]}")
        print("   Trying individual downloads...")
        
        # Method 2: Download individually (slower but more reliable)
        adj_close = pd.DataFrame()
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            try:
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i+1}/{len(tickers)}")
                
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if not data.empty and 'Adj Close' in data.columns:
                    adj_close[ticker] = data['Adj Close']
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                failed_tickers.append(ticker)
                continue
        
        if failed_tickers:
            print(f"\n   ⚠️ Failed to download {len(failed_tickers)} stocks")
            print(f"   First few failed: {failed_tickers[:5]}")
    
    # Clean data
    print("\n🧹 Cleaning data...")
    
    # Remove stocks with too much missing data (>10%)
    missing_pct = adj_close.isnull().sum() / len(adj_close)
    valid_stocks = missing_pct[missing_pct < 0.1].index.tolist()
    
    print(f"   Kept {len(valid_stocks)} stocks with <10% missing data")
    print(f"   Removed {len(adj_close.columns) - len(valid_stocks)} stocks")
    
    # Forward fill remaining missing values
    adj_close_clean = adj_close[valid_stocks].fillna(method='ffill').dropna()
    
    print(f"\n✅ Final dataset: {adj_close_clean.shape[0]} days × {adj_close_clean.shape[1]} stocks")
    print(f"   Date range: {adj_close_clean.index.min().date()} to {adj_close_clean.index.max().date()}")
    
    return adj_close_clean

# Set date range (5 years of data)
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print(f"\n📅 Downloading data from {start_date.date()} to {end_date.date()}")

# Download data
# Start with 50 stocks for testing (much faster!)
# Set sample_size=None to use all stocks (takes longer)
prices = download_stock_data_robust(
    all_tickers, 
    start_date, 
    end_date,
    sample_size=50  # Change to None for all stocks
)

# Save to CSV
prices.to_csv('sp500_prices.csv')
print("\n💾 Data saved to 'sp500_prices.csv'")

# Display sample
print("\n📊 Sample of Price Data:")
print(prices.head())

# ============================================
# STEP 5: CALCULATE RETURNS
# ============================================

# Calculate daily returns
returns = prices.pct_change().dropna()

print("\n📊 Returns Statistics:")
print(f"   Shape: {returns.shape}")
print(f"   Mean daily return: {returns.mean().mean():.4%}")
print(f"   Median daily return: {returns.median().median():.4%}")
print(f"   Std daily return: {returns.std().mean():.4%}")

# Save returns
returns.to_csv('sp500_returns.csv')
print("\n💾 Returns saved to 'sp500_returns.csv'")

# ============================================
# STEP 6: CALCULATE BASIC METRICS
# ============================================

def calculate_stock_metrics(returns):
    """
    Calculate key metrics for each stock
    """
    metrics = pd.DataFrame({
        'Daily_Return': returns.mean(),
        'Annual_Return': returns.mean() * 252,
        'Daily_Volatility': returns.std(),
        'Annual_Volatility': returns.std() * np.sqrt(252),
        'Sharpe_Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    })
    
    # Calculate max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics['Max_Drawdown'] = drawdown.min()
    
    return metrics

metrics = calculate_stock_metrics(returns)

print("\n📊 Stock Metrics Summary:")
print(metrics.describe())

print("\n🏆 Top 10 Stocks by Sharpe Ratio:")
top_10 = metrics.nlargest(10, 'Sharpe_Ratio')[['Annual_Return', 'Annual_Volatility', 'Sharpe_Ratio']]
print(top_10)

# Save metrics
metrics.to_csv('sp500_metrics.csv')
print("\n💾 Metrics saved to 'sp500_metrics.csv'")

# ============================================
# STEP 7: QUICK VISUALIZATIONS
# ============================================

print("\n📊 Generating visualizations...")

# 1. Top performers
cumulative_returns = (1 + returns).cumprod()
top_performers = cumulative_returns.iloc[-1].nlargest(5)

plt.figure(figsize=(14, 8))
for stock in top_performers.index:
    plt.plot(cumulative_returns.index, cumulative_returns[stock], 
             linewidth=2, label=stock)

plt.title('Top 5 Performing Stocks - Cumulative Returns', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n🏆 Top 5 Performers:")
print(top_performers)

# 2. Risk-Return scatter
plt.figure(figsize=(14, 8))
scatter = plt.scatter(
    metrics['Annual_Volatility'], 
    metrics['Annual_Return'],
    c=metrics['Sharpe_Ratio'],
    s=50,
    alpha=0.6,
    cmap='RdYlGn'
)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Annual Volatility (Risk)', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.title('Risk-Return Profile', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# Annotate top performers
top_5 = metrics.nlargest(5, 'Sharpe_Ratio')
for idx, row in top_5.iterrows():
    plt.annotate(
        idx,
        (row['Annual_Volatility'], row['Annual_Return']),
        fontsize=9,
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.tight_layout()
plt.show()

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("✅ SETUP COMPLETE!")
print("="*60)

print("\n📦 Files Created:")
print("   • sp500_prices.csv - Historical price data")
print("   • sp500_returns.csv - Daily returns")
print("   • sp500_metrics.csv - Stock metrics")

print("\n📊 Data Summary:")
print(f"   • {len(prices.columns)} stocks")
print(f"   • {len(prices)} trading days")
print(f"   • Date range: {prices.index.min().date()} to {prices.index.max().date()}")

print("\n📈 Key Statistics:")
print(f"   • Average annual return: {metrics['Annual_Return'].mean():.2%}")
print(f"   • Average volatility: {metrics['Annual_Volatility'].mean():.2%}")
print(f"   • Best Sharpe ratio: {metrics['Sharpe_Ratio'].max():.2f}")

print("\n🎯 Next Steps:")
print("   1. Run portfolio optimization")
print("   2. Build efficient frontier")
print("   3. Calculate VaR and risk metrics")
print("   4. Backtest strategies")
print("   5. Create Streamlit dashboard")

print("\n🚀 You're ready to optimize portfolios!")
print("="*60)
