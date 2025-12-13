"""
Portfolio Optimization - Advanced Risk Analysis & Backtesting
Comprehensive risk metrics, stress testing, and walk-forward backtesting
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker/headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)


class PortfolioRiskAnalyzer:
    """
    Advanced risk analysis for portfolio optimization
    """
    
    def __init__(self, returns, weights=None):
        """
        Parameters:
        -----------
        returns : DataFrame
            Historical returns for assets
        weights : array-like, optional
            Portfolio weights (if None, assumes equal weight)
        """
        self.returns = returns
        if weights is None:
            self.weights = np.ones(len(returns.columns)) / len(returns.columns)
        else:
            self.weights = np.array(weights)
        
        self.portfolio_returns = (returns @ self.weights)
    
    def calculate_var(self, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR)
        
        Returns:
        --------
        float : VaR at specified confidence level
        """
        return np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        
        Returns:
        --------
        float : CVaR at specified confidence level
        """
        var = self.calculate_var(confidence_level)
        return self.portfolio_returns[self.portfolio_returns <= var].mean()
    
    def calculate_maximum_drawdown(self):
        """
        Calculate maximum drawdown
        
        Returns:
        --------
        dict : Max drawdown, duration, and dates
        """
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find duration
        dd_series = drawdown[drawdown < 0]
        if len(dd_series) > 0:
            # Find the longest continuous drawdown period
            dd_groups = (dd_series == 0).cumsum()
            durations = dd_groups.value_counts()
            max_duration = durations.max() if len(durations) > 0 else 0
        else:
            max_duration = 0
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'max_duration_days': max_duration,
            'drawdown_series': drawdown
        }
    
    def calculate_downside_deviation(self, mar=0.0):
        """
        Calculate downside deviation (semi-deviation)
        
        Parameters:
        -----------
        mar : float
            Minimum acceptable return (default 0)
        
        Returns:
        --------
        float : Downside deviation
        """
        downside_returns = self.portfolio_returns[self.portfolio_returns < mar]
        return np.sqrt(np.mean(downside_returns ** 2))
    
    def calculate_sortino_ratio(self, risk_free_rate=0.03, periods=252):
        """
        Calculate Sortino ratio
        
        Returns:
        --------
        float : Sortino ratio
        """
        excess_return = self.portfolio_returns.mean() * periods - risk_free_rate
        downside_dev = self.calculate_downside_deviation() * np.sqrt(periods)
        
        return excess_return / downside_dev if downside_dev > 0 else 0
    
    def calculate_calmar_ratio(self, periods=252):
        """
        Calculate Calmar ratio (Return / Max Drawdown)
        
        Returns:
        --------
        float : Calmar ratio
        """
        annual_return = self.portfolio_returns.mean() * periods
        max_dd = abs(self.calculate_maximum_drawdown()['max_drawdown'])
        
        return annual_return / max_dd if max_dd > 0 else 0
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.03, periods=252):
        """
        Calculate Sharpe ratio
        
        Returns:
        --------
        float : Sharpe ratio
        """
        excess_return = self.portfolio_returns.mean() * periods - risk_free_rate
        volatility = self.portfolio_returns.std() * np.sqrt(periods)
        
        return excess_return / volatility if volatility > 0 else 0
    
    def calculate_tail_ratio(self):
        """
        Calculate tail ratio (95th percentile / 5th percentile)
        
        Returns:
        --------
        float : Tail ratio
        """
        return abs(np.percentile(self.portfolio_returns, 95) / 
                   np.percentile(self.portfolio_returns, 5))
    
    def calculate_skewness(self):
        """Calculate returns skewness"""
        return stats.skew(self.portfolio_returns)
    
    def calculate_kurtosis(self):
        """Calculate returns kurtosis"""
        return stats.kurtosis(self.portfolio_returns)
    
    def get_risk_metrics(self):
        """
        Calculate all risk metrics
        
        Returns:
        --------
        dict : Dictionary of all risk metrics
        """
        dd_info = self.calculate_maximum_drawdown()
        
        metrics = {
            'Annual Return': self.portfolio_returns.mean() * 252,
            'Annual Volatility': self.portfolio_returns.std() * np.sqrt(252),
            'Sharpe Ratio': self.calculate_sharpe_ratio(),
            'Sortino Ratio': self.calculate_sortino_ratio(),
            'Calmar Ratio': self.calculate_calmar_ratio(),
            'Max Drawdown': dd_info['max_drawdown'],
            'Max Drawdown Duration': dd_info['max_duration_days'],
            'VaR (95%)': self.calculate_var(0.95),
            'CVaR (95%)': self.calculate_cvar(0.95),
            'Downside Deviation': self.calculate_downside_deviation() * np.sqrt(252),
            'Skewness': self.calculate_skewness(),
            'Kurtosis': self.calculate_kurtosis(),
            'Tail Ratio': self.calculate_tail_ratio()
        }
        
        return metrics
    
    def plot_risk_analysis(self):
        """Create comprehensive risk analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Cumulative Returns
        cumulative = (1 + self.portfolio_returns).cumprod()
        axes[0, 0].plot(cumulative.index, cumulative.values, linewidth=2)
        axes[0, 0].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown Chart
        dd_info = self.calculate_maximum_drawdown()
        axes[0, 1].fill_between(dd_info['drawdown_series'].index, 
                                dd_info['drawdown_series'].values, 0, 
                                alpha=0.3, color='red')
        axes[0, 1].plot(dd_info['drawdown_series'].index, 
                       dd_info['drawdown_series'].values, 
                       color='red', linewidth=2)
        axes[0, 1].set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Returns Distribution
        axes[1, 0].hist(self.portfolio_returns, bins=50, alpha=0.7, 
                       color='blue', edgecolor='black')
        axes[1, 0].axvline(self.portfolio_returns.mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
        axes[1, 0].axvline(self.calculate_var(0.95), color='orange', 
                          linestyle='--', linewidth=2, label='VaR (95%)')
        axes[1, 0].set_title('Returns Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rolling Volatility
        rolling_vol = self.portfolio_returns.rolling(window=30).std() * np.sqrt(252)
        axes[1, 1].plot(rolling_vol.index, rolling_vol.values, linewidth=2)
        axes[1, 1].set_title('Rolling 30-Day Volatility (Annualized)', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class PortfolioBacktester:
    """
    Walk-forward backtesting for portfolio strategies
    """
    
    def __init__(self, returns_data, lookback_years=3, rebalance_frequency='Q'):
        """
        Parameters:
        -----------
        returns_data : DataFrame
            Historical returns for all assets
        lookback_years : int
            Years of history to use for optimization
        rebalance_frequency : str
            'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
        """
        self.returns_data = returns_data
        self.lookback_days = lookback_years * 252
        self.rebalance_frequency = rebalance_frequency
    
    def optimize_portfolio(self, returns_train, method='max_sharpe', 
                          risk_free_rate=0.03):
        """
        Optimize portfolio weights
        
        Parameters:
        -----------
        returns_train : DataFrame
            Training data returns
        method : str
            'max_sharpe', 'min_variance', 'max_return'
        
        Returns:
        --------
        array : Optimal weights
        """
        n_assets = len(returns_train.columns)
        mean_returns = returns_train.mean() * 252
        cov_matrix = returns_train.cov() * 252
        
        # Objective functions
        def portfolio_volatility(weights):
            return np.sqrt(weights.T @ cov_matrix @ weights)
        
        def neg_sharpe(weights):
            ret = weights @ mean_returns
            vol = np.sqrt(weights.T @ cov_matrix @ weights)
            return -(ret - risk_free_rate) / vol
        
        def neg_return(weights):
            return -(weights @ mean_returns)
        
        # Choose objective
        if method == 'max_sharpe':
            objective = neg_sharpe
        elif method == 'min_variance':
            objective = portfolio_volatility
        elif method == 'max_return':
            objective = neg_return
        else:
            objective = neg_sharpe
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 0.3) for _ in range(n_assets))  # Max 30% per asset
        
        # Initial guess
        w0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else w0
    
    def calculate_transaction_costs(self, old_weights, new_weights, 
                                   portfolio_value, cost_rate=0.001):
        """
        Calculate realistic transaction costs
        
        Parameters:
        -----------
        old_weights : array
            Previous weights
        new_weights : array
            New target weights
        portfolio_value : float
            Current portfolio value
        cost_rate : float
            Transaction cost as % of trade value
        
        Returns:
        --------
        float : Total transaction cost
        """
        turnover = np.sum(np.abs(new_weights - old_weights))
        cost = turnover * portfolio_value * cost_rate
        return cost
    
    def backtest(self, strategy='max_sharpe', include_costs=True):
        """
        Perform walk-forward backtest
        
        Parameters:
        -----------
        strategy : str
            Portfolio optimization strategy
        include_costs : bool
            Whether to include transaction costs
        
        Returns:
        --------
        dict : Backtest results including returns, weights, metrics
        """
        # Generate rebalance dates
        if self.rebalance_frequency == 'M':
            freq = 'MS'  # Month start
        elif self.rebalance_frequency == 'Q':
            freq = 'QS'  # Quarter start
        else:
            freq = 'YS'  # Year start
        
        rebalance_dates = pd.date_range(
            self.returns_data.index[self.lookback_days],
            self.returns_data.index[-1],
            freq=freq
        )
        
        # Initialize tracking
        portfolio_value = 100000  # Start with $100k
        portfolio_values = []
        all_weights = []
        rebalance_info = []
        
        current_weights = None
        
        for i, rebal_date in enumerate(rebalance_dates):
            # Get training data
            train_end_idx = self.returns_data.index.get_loc(rebal_date)
            train_start_idx = max(0, train_end_idx - self.lookback_days)
            
            returns_train = self.returns_data.iloc[train_start_idx:train_end_idx]
            
            # Optimize portfolio
            new_weights = self.optimize_portfolio(returns_train, method=strategy)
            
            # Calculate transaction costs
            if include_costs and current_weights is not None:
                cost = self.calculate_transaction_costs(
                    current_weights, new_weights, portfolio_value
                )
                portfolio_value -= cost
            else:
                cost = 0
            
            # Get next rebalance date
            next_rebal = rebalance_dates[i + 1] if i < len(rebalance_dates) - 1 else self.returns_data.index[-1]
            
            # Calculate returns until next rebalance
            period_returns = self.returns_data[
                (self.returns_data.index >= rebal_date) & 
                (self.returns_data.index < next_rebal)
            ]
            
            # Track portfolio value
            for date, daily_returns in period_returns.iterrows():
                portfolio_return = np.dot(new_weights, daily_returns.values)
                portfolio_value *= (1 + portfolio_return)
                portfolio_values.append({
                    'date': date,
                    'value': portfolio_value,
                    'return': portfolio_return
                })
            
            # Store rebalance info
            rebalance_info.append({
                'date': rebal_date,
                'weights': new_weights.copy(),
                'cost': cost,
                'portfolio_value': portfolio_value
            })
            
            all_weights.append(new_weights)
            current_weights = new_weights.copy()
        
        # Create results DataFrame
        results_df = pd.DataFrame(portfolio_values)
        results_df.set_index('date', inplace=True)
        
        # Calculate metrics
        returns_series = pd.Series(
            [x['return'] for x in portfolio_values],
            index=[x['date'] for x in portfolio_values]
        )
        
        risk_analyzer = PortfolioRiskAnalyzer(
            self.returns_data.loc[results_df.index],
            weights=current_weights
        )
        risk_analyzer.portfolio_returns = returns_series
        
        metrics = risk_analyzer.get_risk_metrics()
        
        # Add total return
        metrics['Total Return'] = (portfolio_value - 100000) / 100000
        metrics['CAGR'] = ((portfolio_value / 100000) ** (252 / len(results_df)) - 1)
        
        return {
            'portfolio_values': results_df,
            'returns': returns_series,
            'weights_history': all_weights,
            'rebalance_info': rebalance_info,
            'metrics': metrics,
            'final_weights': current_weights
        }
    
    def compare_strategies(self, strategies=['max_sharpe', 'min_variance']):
        """
        Compare multiple strategies
        
        Returns:
        --------
        dict : Results for each strategy
        """
        results = {}
        
        for strategy in strategies:
            print(f"\nBacktesting {strategy}...")
            results[strategy] = self.backtest(strategy=strategy)
        
        return results
    
    def plot_backtest_results(self, results_dict):
        """Plot comparison of multiple strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Cumulative Returns
        for strategy, results in results_dict.items():
            axes[0, 0].plot(results['portfolio_values'].index,
                          results['portfolio_values']['value'],
                          label=strategy, linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdowns
        for strategy, results in results_dict.items():
            cumulative = results['portfolio_values']['value'] / 100000
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            axes[0, 1].plot(drawdown.index, drawdown.values, 
                          label=strategy, linewidth=2)
        axes[0, 1].set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Returns Distribution
        for strategy, results in results_dict.items():
            axes[1, 0].hist(results['returns'], bins=50, alpha=0.5, 
                          label=strategy)
        axes[1, 0].set_title('Returns Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Metrics Comparison
        metrics_to_plot = ['Sharpe Ratio', 'Max Drawdown', 'Total Return']
        strategies_list = list(results_dict.keys())
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        for i, strategy in enumerate(strategies_list):
            values = [results_dict[strategy]['metrics'][m] for m in metrics_to_plot]
            # Normalize for display
            values_norm = [v * 100 if 'Return' in m or 'Drawdown' in m else v 
                          for v, m in zip(values, metrics_to_plot)]
            axes[1, 1].bar(x + i * width, values_norm, width, label=strategy)
        
        axes[1, 1].set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x + width / 2)
        axes[1, 1].set_xticklabels(metrics_to_plot, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("PORTFOLIO RISK ANALYSIS & BACKTESTING")
    print("="*70)
    
    # Download sample data
    print("\n📊 Downloading sample data...")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'UNH']
    data = yf.download(tickers, start='2018-01-01', end='2024-01-01', progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    print(f"Downloaded {len(returns)} days of data for {len(tickers)} assets")
    
    # Risk Analysis
    print("\n" + "="*70)
    print("RISK ANALYSIS - Equal Weight Portfolio")
    print("="*70)
    
    risk_analyzer = PortfolioRiskAnalyzer(returns)
    metrics = risk_analyzer.get_risk_metrics()
    
    for metric, value in metrics.items():
        if 'Ratio' in metric or 'Return' in metric or 'Volatility' in metric:
            print(f"{metric:.<40} {value:.4f}")
        else:
            print(f"{metric:.<40} {value:.6f}")
    
    # Create risk plots
    print("\n📊 Creating risk analysis plots...")
    fig = risk_analyzer.plot_risk_analysis()
    plt.savefig('risk_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved risk_analysis.png")
    
    # Backtesting
    print("\n" + "="*70)
    print("WALK-FORWARD BACKTESTING")
    print("="*70)
    
    backtester = PortfolioBacktester(returns, lookback_years=2, rebalance_frequency='Q')
    
    # Compare strategies
    strategies = {
        'Max Sharpe': 'max_sharpe',
        'Min Variance': 'min_variance'
    }
    
    results = backtester.compare_strategies(list(strategies.values()))
    
    # Print results
    print("\n📈 BACKTEST RESULTS:")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        name: res['metrics']
        for name, res in results.items()
    }).T
    
    print(comparison_df[['Total Return', 'CAGR', 'Sharpe Ratio', 'Max Drawdown', 
                         'Sortino Ratio', 'Calmar Ratio']])
    
    # Create comparison plots
    print("\n📊 Creating backtest comparison plots...")
    fig = backtester.plot_backtest_results(results)
    plt.savefig('backtest_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved backtest_comparison.png")
    
    print("\n✅ Analysis complete!")