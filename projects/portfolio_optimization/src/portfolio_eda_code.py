# ============================================
# PORTFOLIO EXPLORATORY DATA ANALYSIS
# Run this after getting your data
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PORTFOLIO EXPLORATORY DATA ANALYSIS")
print("="*70)

# ============================================
# LOAD DATA
# ============================================

print("\n📊 Loading data...")

prices = pd.read_csv('sp500_prices.csv', index_col=0, parse_dates=True)
returns = pd.read_csv('sp500_returns.csv', index_col=0, parse_dates=True)
metrics = pd.read_csv('sp500_metrics.csv', index_col=0)

print(f"✅ Loaded:")
print(f"   Prices: {prices.shape}")
print(f"   Returns: {returns.shape}")
print(f"   Metrics: {metrics.shape}")

# ============================================
# 1. CORRELATION ANALYSIS
# ============================================

print("\n" + "="*70)
print("1. CORRELATION ANALYSIS")
print("="*70)

# Calculate correlation matrix
corr_matrix = returns.corr()

print(f"\n📊 Correlation Statistics:")
print(f"   Average correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
print(f"   Max correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")
print(f"   Min correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.3f}")

# Heatmap (top 20 stocks)
top_20 = metrics.nlargest(20, 'Sharpe_Ratio').index
corr_sample = corr_matrix.loc[top_20, top_20]

plt.figure(figsize=(14, 12))
sns.heatmap(corr_sample, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Top 20 Stocks by Sharpe Ratio', 
          fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("💾 Saved: correlation_heatmap.png")

# Find highly correlated pairs
corr_pairs = corr_matrix.unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0]
high_corr = corr_pairs[corr_pairs > 0.8].sort_values(ascending=False)

print(f"\n🔗 Highly Correlated Pairs (>0.8):")
if len(high_corr) > 0:
    print(high_corr.head(10))
else:
    print("   No pairs with correlation >0.8")

# ============================================
# 2. DISTRIBUTION ANALYSIS
# ============================================

print("\n" + "="*70)
print("2. RETURN DISTRIBUTION ANALYSIS")
print("="*70)

# Overall portfolio statistics
print("\n📊 Return Distribution:")
print(f"   Mean daily return: {returns.mean().mean():.4%}")
print(f"   Median daily return: {returns.median().median():.4%}")
print(f"   Std daily return: {returns.std().mean():.4%}")
print(f"   Skewness: {returns.skew().mean():.2f}")
print(f"   Kurtosis: {returns.kurtosis().mean():.2f}")

# Distribution plots for top 5 stocks
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

top_5 = metrics.nlargest(5, 'Sharpe_Ratio').index

for i, stock in enumerate(top_5):
    returns[stock].hist(bins=50, ax=axes[i], alpha=0.7, edgecolor='black')
    axes[i].axvline(returns[stock].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {returns[stock].mean():.4f}')
    axes[i].set_title(f'{stock} - Daily Returns', fontweight='bold')
    axes[i].set_xlabel('Return')
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Q-Q plot for normality check
stock = top_5[0]
stats.probplot(returns[stock], dist="norm", plot=axes[5])
axes[5].set_title(f'{stock} - Q-Q Plot (Normality Check)', fontweight='bold')
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('return_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("💾 Saved: return_distributions.png")

# ============================================
# 3. TIME SERIES ANALYSIS
# ============================================

print("\n" + "="*70)
print("3. TIME SERIES PATTERNS")
print("="*70)

# Rolling volatility
window = 30  # 30-day rolling window
rolling_vol = returns.std(axis=1).rolling(window=window).mean() * np.sqrt(252)

plt.figure(figsize=(14, 6))
plt.plot(rolling_vol.index, rolling_vol, linewidth=2, color='red')
plt.title('30-Day Rolling Portfolio Volatility (Annualized)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rolling_volatility.png', dpi=300, bbox_inches='tight')
plt.show()

print("💾 Saved: rolling_volatility.png")

# Identify high volatility periods
high_vol_periods = rolling_vol[rolling_vol > rolling_vol.quantile(0.9)]
print(f"\n⚠️ High Volatility Periods (>90th percentile):")
print(f"   Dates: {len(high_vol_periods)} days")
print(f"   Max volatility: {rolling_vol.max():.2%}")
print(f"   Date of max: {rolling_vol.idxmax().date()}")

# ============================================
# 4. PERFORMANCE CLUSTERING
# ============================================

print("\n" + "="*70)
print("4. PERFORMANCE CLUSTERING")
print("="*70)

# Categorize stocks
metrics['Risk_Category'] = pd.cut(
    metrics['Annual_Volatility'], 
    bins=[0, 0.15, 0.25, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

metrics['Return_Category'] = pd.cut(
    metrics['Annual_Return'],
    bins=[-1, 0, 0.15, 1.0],
    labels=['Negative', 'Low Return', 'High Return']
)

# Create performance matrix
performance_matrix = pd.crosstab(
    metrics['Risk_Category'], 
    metrics['Return_Category']
)

print("\n📊 Stock Distribution by Risk-Return Profile:")
print(performance_matrix)

# Best quadrant: High Return + Low/Medium Risk
best_stocks = metrics[
    (metrics['Annual_Return'] > 0.15) & 
    (metrics['Annual_Volatility'] < 0.25)
].sort_values('Sharpe_Ratio', ascending=False)

print(f"\n🏆 Best Risk-Adjusted Stocks (High Return, Low-Med Risk):")
print(f"   Count: {len(best_stocks)}")
if len(best_stocks) > 0:
    print(best_stocks.head(10)[['Annual_Return', 'Annual_Volatility', 'Sharpe_Ratio']])

# ============================================
# 5. DRAWDOWN ANALYSIS
# ============================================

print("\n" + "="*70)
print("5. DRAWDOWN ANALYSIS")
print("="*70)

# Calculate portfolio drawdown (equal weighted)
portfolio_returns = returns.mean(axis=1)
cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max

max_dd = drawdown.min()
max_dd_date = drawdown.idxmin()

print(f"\n📉 Equal-Weight Portfolio Drawdown:")
print(f"   Maximum drawdown: {max_dd:.2%}")
print(f"   Date: {max_dd_date.date()}")

# Plot drawdown
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Cumulative returns
axes[0].plot(cumulative.index, cumulative, linewidth=2, color='blue', label='Portfolio')
axes[0].plot(running_max.index, running_max, linewidth=2, color='red', 
             linestyle='--', alpha=0.7, label='Peak')
axes[0].set_ylabel('Cumulative Return', fontsize=12)
axes[0].set_title('Equal-Weight Portfolio Performance', fontsize=16, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Drawdown
axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.5, color='red')
axes[1].axhline(max_dd, color='darkred', linestyle='--', linewidth=2)
axes[1].set_ylabel('Drawdown', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_title(f'Drawdown (Max: {max_dd:.2%})', fontsize=16, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('portfolio_drawdown.png', dpi=300, bbox_inches='tight')
plt.show()

print("💾 Saved: portfolio_drawdown.png")

# ============================================
# 6. INTERACTIVE EFFICIENT FRONTIER PREVIEW
# ============================================

print("\n" + "="*70)
print("6. RISK-RETURN LANDSCAPE (INTERACTIVE)")
print("="*70)

# Create interactive scatter
fig = px.scatter(
    metrics,
    x='Annual_Volatility',
    y='Annual_Return',
    color='Sharpe_Ratio',
    size='Sharpe_Ratio',
    hover_data=['Max_Drawdown'],
    labels={
        'Annual_Volatility': 'Annual Volatility',
        'Annual_Return': 'Annual Return',
        'Sharpe_Ratio': 'Sharpe Ratio'
    },
    title='Risk-Return Landscape - All Stocks',
    color_continuous_scale='RdYlGn'
)

fig.update_layout(
    width=1000,
    height=700,
    hovermode='closest'
)

fig.show()

print("📊 Interactive plot displayed")

# ============================================
# 7. SUMMARY STATISTICS
# ============================================

print("\n" + "="*70)
print("EXPLORATORY ANALYSIS SUMMARY")
print("="*70)

summary = {
    'Total Stocks': len(metrics),
    'Avg Annual Return': f"{metrics['Annual_Return'].mean():.2%}",
    'Avg Volatility': f"{metrics['Annual_Volatility'].mean():.2%}",
    'Avg Sharpe Ratio': f"{metrics['Sharpe_Ratio'].mean():.2f}",
    'Best Sharpe': f"{metrics['Sharpe_Ratio'].max():.2f}",
    'Avg Correlation': f"{corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}",
    'Portfolio Max DD': f"{max_dd:.2%}",
    'High Risk-Adj Stocks': len(best_stocks)
}

print("\n📊 Key Findings:")
for key, value in summary.items():
    print(f"   • {key}: {value}")

print("\n📁 Files Created:")
print("   ✓ correlation_heatmap.png")
print("   ✓ return_distributions.png")
print("   ✓ rolling_volatility.png")
print("   ✓ portfolio_drawdown.png")

print("\n🎯 Key Insights for Optimization:")
print(f"   1. Average stock correlation is {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.1%}")
print(f"      → {'Good' if corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean() < 0.5 else 'Moderate'} diversification potential")
print(f"   2. {len(best_stocks)} stocks have high returns with manageable risk")
print(f"      → These are prime candidates for portfolio")
print(f"   3. Equal-weight portfolio had {max_dd:.1%} max drawdown")
print(f"      → Optimization should aim to reduce this")

print("\n✅ EDA COMPLETE! Ready for optimization!")
print("="*70)