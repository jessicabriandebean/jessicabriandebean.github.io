# 🚀 Portfolio Optimization Project - Complete Setup Guide

A comprehensive guide to setting up the S&P 500 Portfolio Optimization project with risk analysis and backtesting capabilities.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Installation Steps](#installation-steps)
4. [Running the Project](#running-the-project)
5. [Understanding the Components](#understanding-the-components)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Required Software
- **Python 3.8+** (3.9 or 3.10 recommended)
- **pip** (Python package manager)
- **Git** (for version control)
- **Jupyter Notebook** (for analysis)

### Check Your Python Version
```bash
python --version
# or
python3 --version
```

Should output: `Python 3.8.x` or higher

---

## 📁 Project Structure

Create the following directory structure:

```
portfolio-optimization/
│
├── portfolio_optimizer.py           # Main optimization code
├── portfolio_risk_backtest.py       # Risk analysis & backtesting
├── streamlit_app.py                 # Interactive web app
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── .gitignore                       # Git ignore file
│
├── notebooks/
│   └── analysis.ipynb              # Jupyter analysis notebook
│
├── data/
│   └── (downloaded price data)
│
├── results/
│   ├── risk_analysis.png
│   ├── backtest_comparison.png
│   └── efficient_frontier.png
│
└── tests/
    └── test_optimizer.py           # Unit tests
```

---

## 📥 Installation Steps

### Step 1: Create Project Directory

```bash
# Create main project folder
mkdir portfolio-optimization
cd portfolio-optimization

# Create subdirectories
mkdir notebooks data results tests
```

### Step 2: Create Virtual Environment

**Using venv (Python built-in):**
```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n portfolio-opt python=3.9
conda activate portfolio-opt
```

### Step 3: Create requirements.txt

Create a file named `requirements.txt` with the following content:

```txt
# Core dependencies
numpy==1.24.3
pandas==2.0.3
scipy==1.11.2

# Data fetching
yfinance==0.2.28

# Optimization
cvxpy==1.3.2
PyPortfolioOpt==1.5.5

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0

# Web application
streamlit==1.28.0

# Jupyter notebook
jupyter==1.0.0
notebook==7.0.3
ipykernel==6.25.2

# Testing
pytest==7.4.2
pytest-cov==4.1.0

# Utilities
python-dateutil==2.8.2
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This may take 3-5 minutes. You should see output like:
```
Collecting numpy==1.24.3
  Downloading numpy-1.24.3-cp39-cp39-...
Successfully installed numpy-1.24.3 pandas-2.0.3 ...
```

### Step 5: Verify Installation

```bash
python -c "import numpy, pandas, yfinance, scipy, matplotlib; print('✅ All packages installed successfully!')"
```

### Step 6: Create Project Files

Copy the code into these files:

1. **`portfolio_risk_backtest.py`** - The risk analysis and backtesting code
2. **`portfolio_optimizer.py`** - Your main optimization code
3. **`streamlit_app.py`** - Web interface (optional)

### Step 7: Create .gitignore

Create `.gitignore` file:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# Data files
data/*.csv
data/*.pkl

# Results
results/*.png
results/*.pdf

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
htmlcov/
.coverage
```

---

## 🎯 Running the Project

### Option 1: Quick Test Run

Create a test script `test_run.py`:

```python
"""
Quick test of portfolio optimization system
"""

from portfolio_risk_backtest import PortfolioRiskAnalyzer, PortfolioBacktester
import yfinance as yf
import pandas as pd

print("="*70)
print("PORTFOLIO OPTIMIZATION - QUICK TEST")
print("="*70)

# Download sample data
print("\n📊 Downloading data...")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
data = yf.download(tickers, start='2020-01-01', end='2024-01-01', progress=False)['Adj Close']
returns = data.pct_change().dropna()

print(f"✅ Downloaded {len(returns)} days of data for {len(tickers)} assets")

# Risk Analysis
print("\n" + "="*70)
print("RISK ANALYSIS")
print("="*70)

analyzer = PortfolioRiskAnalyzer(returns)
metrics = analyzer.get_risk_metrics()

for metric, value in metrics.items():
    print(f"{metric:.<40} {value:.4f}")

# Create plots
print("\n📊 Creating risk analysis plots...")
fig = analyzer.plot_risk_analysis()
import matplotlib.pyplot as plt
plt.savefig('results/risk_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved results/risk_analysis.png")

# Backtesting
print("\n" + "="*70)
print("BACKTESTING")
print("="*70)

backtester = PortfolioBacktester(returns, lookback_years=2, rebalance_frequency='Q')
results = backtester.backtest(strategy='max_sharpe', include_costs=True)

print(f"\nTotal Return: {results['metrics']['Total Return']:.2%}")
print(f"CAGR: {results['metrics']['CAGR']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['Sharpe Ratio']:.4f}")
print(f"Max Drawdown: {results['metrics']['Max Drawdown']:.2%}")

print("\n✅ Test complete!")
```

Run it:
```bash
python test_run.py
```

### Option 2: Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder
# Create new notebook: analysis.ipynb
```

In the notebook:

```python
# Cell 1: Setup
%load_ext autoreload
%autoreload 2

import sys
sys.path.append('..')

from portfolio_risk_backtest import PortfolioRiskAnalyzer, PortfolioBacktester
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

print("✅ Setup complete!")
```

```python
# Cell 2: Download Data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'UNH']
data = yf.download(tickers, start='2018-01-01', end='2024-01-01')['Adj Close']
returns = data.pct_change().dropna()

print(f"Data shape: {returns.shape}")
print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
```

```python
# Cell 3: Risk Analysis
analyzer = PortfolioRiskAnalyzer(returns)
metrics = analyzer.get_risk_metrics()

# Display metrics
pd.DataFrame(metrics, index=['Value']).T
```

```python
# Cell 4: Visualize Risk
fig = analyzer.plot_risk_analysis()
plt.show()
```

```python
# Cell 5: Backtest
backtester = PortfolioBacktester(returns, lookback_years=3, rebalance_frequency='Q')
results = backtester.backtest(strategy='max_sharpe', include_costs=True)

# Display results
print(f"Total Return: {results['metrics']['Total Return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['Sharpe Ratio']:.4f}")
```

```python
# Cell 6: Compare Strategies
comparison = backtester.compare_strategies(['max_sharpe', 'min_variance'])

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    name: res['metrics']
    for name, res in comparison.items()
}).T

comparison_df[['Total Return', 'CAGR', 'Sharpe Ratio', 'Max Drawdown']]
```

```python
# Cell 7: Plot Comparison
fig = backtester.plot_backtest_results(comparison)
plt.show()
```

### Option 3: Streamlit Web App

If you have `streamlit_app.py`, run:

```bash
streamlit run streamlit_app.py
```

Access at `http://localhost:8501`

---

## 🐳 Docker Setup (Recommended for Production)

### Step 1: Create Dockerfile

Create a file named `Dockerfile` in your project root:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for outputs
RUN mkdir -p data results notebooks

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit (if using)
EXPOSE 8501

# Default command (can be overridden)
CMD ["python", "test_run.py"]
```

### Step 2: Create docker-compose.yml

Create `docker-compose.yml` for easier management:

```yaml
version: '3.8'

services:
  portfolio-optimizer:
    build: .
    container_name: portfolio-optimizer
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./notebooks:/app/notebooks
    environment:
      - PYTHONUNBUFFERED=1
    command: python test_run.py
    
  portfolio-streamlit:
    build: .
    container_name: portfolio-streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    command: streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
    
  jupyter:
    build: .
    container_name: portfolio-jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./:/app
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
```

### Step 3: Create .dockerignore

Create `.dockerignore` to exclude unnecessary files:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
venv/
env/
ENV/

# Git
.git
.gitignore

# IDE
.vscode/
.idea/

# Data and results (will be mounted as volumes)
data/*
results/*

# Jupyter
.ipynb_checkpoints
notebooks/.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# Docker
Dockerfile
docker-compose.yml
.dockerignore
```

### Step 4: Build and Run with Docker

**Build the Docker image:**
```bash
docker build -t portfolio-optimizer .
```

**Run risk analysis:**
```bash
docker run -v $(pwd)/results:/app/results portfolio-optimizer python test_run.py
```

**Run with docker-compose:**
```bash
# Run the optimizer
docker-compose run portfolio-optimizer

# Run Streamlit app
docker-compose up portfolio-streamlit

# Run Jupyter notebook
docker-compose up jupyter

# Run all services
docker-compose up
```

### Step 5: Docker Commands Reference

```bash
# Build image
docker build -t portfolio-optimizer .

# Run interactive shell
docker run -it portfolio-optimizer /bin/bash

# Run specific script
docker run -v $(pwd)/results:/app/results portfolio-optimizer python test_run.py

# View logs
docker logs portfolio-optimizer

# Stop containers
docker-compose down

# Rebuild and restart
docker-compose up --build
```

### Step 6: Docker Best Practices for This Project

**Mount volumes for data persistence:**
```bash
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/results:/app/results \
           portfolio-optimizer python test_run.py
```

**Run Jupyter in Docker:**
```bash
docker run -p 8888:8888 \
           -v $(pwd):/app \
           portfolio-optimizer \
           jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Run Streamlit in Docker:**
```bash
docker run -p 8501:8501 \
           -v $(pwd)/results:/app/results \
           portfolio-optimizer \
           streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

### Step 7: Update test_run.py for Docker

Modify `test_run.py` to save plots in Docker-friendly way:

```python
"""
Quick test of portfolio optimization system (Docker-compatible)
"""

from portfolio_risk_backtest import PortfolioRiskAnalyzer, PortfolioBacktester
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import os

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

print("="*70)
print("PORTFOLIO OPTIMIZATION - QUICK TEST (DOCKER)")
print("="*70)

# Download sample data
print("\n📊 Downloading data...")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
data = yf.download(tickers, start='2020-01-01', end='2024-01-01', progress=False)['Adj Close']
returns = data.pct_change().dropna()

print(f"✅ Downloaded {len(returns)} days of data for {len(tickers)} assets")

# Risk Analysis
print("\n" + "="*70)
print("RISK ANALYSIS")
print("="*70)

analyzer = PortfolioRiskAnalyzer(returns)
metrics = analyzer.get_risk_metrics()

for metric, value in metrics.items():
    print(f"{metric:.<40} {value:.4f}")

# Create plots
print("\n📊 Creating risk analysis plots...")
fig = analyzer.plot_risk_analysis()
plt.savefig('/app/results/risk_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved /app/results/risk_analysis.png")

# Backtesting
print("\n" + "="*70)
print("BACKTESTING")
print("="*70)

backtester = PortfolioBacktester(returns, lookback_years=2, rebalance_frequency='Q')
results = backtester.backtest(strategy='max_sharpe', include_costs=True)

print(f"\nTotal Return: {results['metrics']['Total Return']:.2%}")
print(f"CAGR: {results['metrics']['CAGR']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['Sharpe Ratio']:.4f}")
print(f"Max Drawdown: {results['metrics']['Max Drawdown']:.2%}")

# Save results to CSV
print("\n💾 Saving results...")
results['portfolio_values'].to_csv('/app/results/portfolio_values.csv')
pd.DataFrame([results['metrics']]).to_csv('/app/results/metrics.csv')
print("✅ Saved results to CSV files")

print("\n✅ Test complete!")
```

### Step 8: Environment Variables for Docker

Create `.env` file for configuration:

```bash
# Python environment
PYTHONUNBUFFERED=1

# Data settings
START_DATE=2018-01-01
END_DATE=2024-01-01

# Portfolio settings
LOOKBACK_YEARS=3
REBALANCE_FREQ=Q
RISK_FREE_RATE=0.03

# File paths
DATA_DIR=/app/data
RESULTS_DIR=/app/results
```

Use in docker-compose.yml:
```yaml
services:
  portfolio-optimizer:
    build: .
    env_file: .env
    volumes:
      - ./data:${DATA_DIR}
      - ./results:${RESULTS_DIR}
```

### Step 9: Multi-stage Docker Build (Optimized)

For production, use multi-stage build:

```dockerfile
# Build stage
FROM python:3.9 as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data results notebooks

# Set PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Run
CMD ["python", "test_run.py"]
```

This creates a smaller final image (~400MB vs ~1GB)

---

## 🧩 Understanding the Components

### 1. PortfolioRiskAnalyzer

**Purpose:** Calculate comprehensive risk metrics for a portfolio

**Key Methods:**
```python
analyzer = PortfolioRiskAnalyzer(returns, weights)