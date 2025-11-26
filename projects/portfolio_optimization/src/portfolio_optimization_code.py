# ============================================
# PORTFOLIO OPTIMIZATION
# Modern Portfolio Theory Implementation
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PORTFOLIO OPTIMIZATION")
print("="*70)

# ============================================
# LOAD DATA
# ============================================

print("\n📊 Loading data...")

prices = pd.read_csv('sp500_prices.csv', index_col=0, parse_dates=True)
returns = pd.read_csv('sp500_returns.csv', index_col=0, parse_dates=True)
metrics = pd.read_csv('sp500_metrics.csv', index_col=0)

print(f"✅ Loaded {len(returns.columns)} stocks")

# ============================================
# PORTFOLIO OPTIMIZER CLASS
# ============================================

class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory
    """
    
    def __init__(self, returns, risk_free_rate=0.02):
        self.returns = returns
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        
    def portfolio_stats(self, weights):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        returns = np.dot(weights, self.mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (returns - self.risk_free_rate) / volatility
        return returns, volatility, sharpe
    
    def negative_sharpe(self, weights):
        """Negative Sharpe for minimization"""
        return -self.portfolio_stats(weights)[2]
    
    def portfolio_volatility(self, weights):
        """Portfolio volatility"""
        return self.portfolio_stats(weights)[1]
    
    def max_sharpe_ratio(self):
        """Find portfolio with maximum Sharpe ratio"""
        print("\n🎯 Optimizing for Maximum Sharpe Ratio...")
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 0.10) for _ in range(self.n_assets))  # Max 10% per stock
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            self.negative_sharpe,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            returns, vol, sharpe = self.portfolio_stats(result.x)
            print(f"✅ Optimization successful!")
            print(f"   Expected Return: {returns:.2%}")
            print(f"   Volatility: {vol:.2%}")
            print(f"   Sharpe Ratio: {sharpe:.2f}")
        else:
            print(f"⚠️ Optimization failed: {result.message}")
        
        return result
    
    def min_volatility(self):
        """Find minimum volatility portfolio"""
        print("\n🎯 Optimizing for Minimum Volatility...")
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 0.10) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            self.portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            returns, vol, sharpe = self.portfolio_stats(result.x)
            print(f"✅ Optimization successful!")
            print(f"   Expected Return: {returns:.2%}")
            print(f"   Volatility: {vol:.2%}")
            print(f"   Sharpe Ratio: {sharpe:.2f}")
        
        return result
    
    def efficient_frontier(self, n_portfolios=50):
        """Generate efficient frontier"""
        print(f"\n📊 Generating efficient frontier ({n_portfolios} portfolios)...")
        
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target in target_returns:
            # Minimize volatility for target return
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, self.mean_returns) - target}
            ]
            bounds = tuple((0, 0.10) for _ in range(self.n_assets))
            initial_guess = np.array([1/self.n_assets] * self.n_assets)
            
            result = minimize(
                self.portfolio_volatility,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                ret, vol, sharpe = self.portfolio_stats(result.x)
                efficient_portfolios.append({
                    'Return': ret,
                    'Volatility': vol,
                    'Sharpe': sharpe
                })
        
        print(f"✅ Generated {len(efficient_portfolios)} efficient portfolios")
        return pd.DataFrame(efficient_portfolios)
    
    def get_holdings(self, weights, top_n=15):
        """Get top holdings from portfolio weights"""
        holdings = pd.DataFrame({
            'Stock': self.returns.columns,
            'Weight': weights
        }).sort_values('Weight', ascending=False)
        
        holdings = holdings[holdings['Weight'] > 0.001]  # Filter tiny positions
        holdings['Weight_Pct'] = holdings['Weight'] * 100
        
        return holdings.head(top_n)

# ============================================
# RUN OPTIMIZATIONS
# ============================================

# Initialize optimizer
optimizer = PortfolioOptimizer(returns, risk_free_rate=0.02)

# Store results
results = {}

# 1. Maximum Sharpe Ratio
max_sharpe = optimizer.max_sharpe_ratio()
results['Max Sharpe'] = {
    'weights': max_sharpe.x,
    'stats': optimizer.portfolio_stats(max_sharpe.x)
}

# 2. Minimum Volatility
min_vol = optimizer.min_volatility()
results['Min Volatility'] = {
    'weights': min_vol.x,
    'stats': optimizer.portfolio_stats(min_vol.x)
}

# 3. Equal Weight (benchmark)
equal_weights = np.array([1/optimizer.n_assets] * optimizer.n_assets)
results['Equal Weight'] = {
    'weights': equal_weights,
    'stats': optimizer.portfolio_stats(equal_weights)
}

# ============================================
# COMPARE STRATEGIES
# ============================================

print("\n" + "="*70)
print("STRATEGY COMPARISON")
print("="*70)

comparison = pd.DataFrame({
    'Strategy': list(results.keys()),
    'Return': [results[s]['stats'][0] for s in results.keys()],
    'Volatility': [results[s]['stats'][1] for s in results.keys()],
    'Sharpe': [results[s]['stats'][2] for s in results.keys()]
})

print("\n📊 Performance Comparison:")
print(comparison.to_string(index=False))

best_sharpe = comparison.loc[comparison['Sharpe'].idxmax(), 'Strategy']
print(f"\n🏆 Best Strategy: {best_sharpe}")

# ============================================
# VISUALIZE EFFICIENT FRONTIER
# ============================================

print("\n📊 Creating efficient frontier visualization...")

# Generate frontier
efficient_frontier = optimizer.efficient_frontier(n_portfolios=50)

# Create plot
fig = go.Figure()

# Efficient frontier
fig.add_trace(go.Scatter(
    x=efficient_frontier['Volatility'] * 100,
    y=efficient_frontier['Return'] * 100,
    mode='lines',
    name='Efficient Frontier',
    line=dict(color='blue', width=3)
))

# Individual stocks
fig.add_trace(go.Scatter(
    x=metrics['Annual_Volatility'] * 100,
    y=metrics['Annual_Return'] * 100,
    mode='markers',
    name='Individual Stocks',
    marker=dict(size=6, color='gray', opacity=0.5),
    text=metrics.index,
    hovertemplate='<b>%{text}</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%'
))

# Max Sharpe portfolio
fig.add_trace(go.Scatter(
    x=[results['Max Sharpe']['stats'][1] * 100],
    y=[results['Max Sharpe']['stats'][0] * 100],
    mode='markers',
    name=f"Max Sharpe ({results['Max Sharpe']['stats'][2]:.2f})",
    marker=dict(size=15, color='red', symbol='star')
))

# Min Volatility portfolio
fig.add_trace(go.Scatter(
    x=[results['Min Volatility']['stats'][1] * 100],
    y=[results['Min Volatility']['stats'][0] * 100],
    mode='markers',
    name=f"Min Volatility ({results['Min Volatility']['stats'][2]:.2f})",
    marker=dict(size=15, color='green', symbol='star')
))

# Equal Weight portfolio
fig.add_trace(go.Scatter(
    x=[results['Equal Weight']['stats'][1] * 100],
    y=[results['Equal Weight']['stats'][0] * 100],
    mode='markers',
    name=f"Equal Weight ({results['Equal Weight']['stats'][2]:.2f})",
    marker=dict(size=15, color='orange', symbol='diamond')
))

fig.update_layout(
    title='Efficient Frontier - Portfolio Optimization',
    xaxis_title='Annual Volatility (%)',
    yaxis_title='Annual Return (%)',
    hovermode='closest',
    height=700,
    width=1000,
    template='plotly_white'
)

fig.show()
fig.write_html('efficient_frontier.html')

print("💾 Saved: efficient_frontier.html")

# ============================================
# DISPLAY TOP HOLDINGS
# ============================================

print("\n" + "="*70)
print("PORTFOLIO HOLDINGS")
print("="*70)

for strategy in ['Max Sharpe', 'Min Volatility']:
    print(f"\n🎯 {strategy} Portfolio - Top 15 Holdings:")
    holdings = optimizer.get_holdings(results[strategy]['weights'])
    print(holdings[['Stock', 'Weight_Pct']].to_string(index=False))
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    holdings.plot(kind='barh', x='Stock', y='Weight_Pct', ax=ax, 
                  color='steelblue', legend=False)
    ax.set_xlabel('Weight (%)', fontsize=12)
    ax.set_title(f'{strategy} Portfolio - Top Holdings', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{strategy.lower().replace(" ", "_")}_holdings.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Saved: {strategy.lower().replace(' ', '_')}_holdings.png")

# ============================================
# SAVE RESULTS
# ============================================

# Save optimal weights
optimal_weights = pd.DataFrame({
    'Stock': optimizer.returns.columns,
    'Max_Sharpe_Weight': results['Max Sharpe']['weights'],
    'Min_Vol_Weight': results['Min Volatility']['weights'],
    'Equal_Weight': results['Equal Weight']['weights']
})

optimal_weights.to_csv('optimal_weights.csv', index=False)
print("\n💾 Saved: optimal_weights.csv")

# Save comparison
comparison.to_csv('strategy_comparison.csv', index=False)
print("💾 Saved: strategy_comparison.csv")

print("\n" + "="*70)
print("✅ OPTIMIZATION COMPLETE!")
print("="*70)

print("\n📁 Files Created:")
print("   ✓ efficient_frontier.html (interactive)")
print("   ✓ max_sharpe_holdings.png")
print("   ✓ min_volatility_holdings.png")
print("   ✓ optimal_weights.csv")
print("   ✓ strategy_comparison.csv")

print("\n🎯 Key Results:")
print(f"   • Max Sharpe Ratio: {results['Max Sharpe']['stats'][2]:.2f}")
print(f"   • Improvement vs Equal Weight: {((results['Max Sharpe']['stats'][2] / results['Equal Weight']['stats'][2]) - 1) * 100:.1f}%")
print(f"   • Optimal Return: {results['Max Sharpe']['stats'][0]:.2%}")
print(f"   • Optimal Volatility: {results['Max Sharpe']['stats'][1]:.2%}")

print("\n✅ Ready for risk analysis and backtesting!")
print("="*70)