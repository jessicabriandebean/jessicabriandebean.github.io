# ============================================
# COMPLETE ECONOMIC FORECASTING IMPLEMENTATION
# Step-by-step: Data Processing → Modeling → Evaluation
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Statistical models
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Prophet
from prophet import Prophet

# LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

# ============================================
# STEP 1: HANDLE MISSING VALUES
# ============================================

def handle_missing_values(df, method='forward_fill'):
    """
    Handle missing values in time series data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with datetime index
    method : str
        'forward_fill', 'interpolate', or 'hybrid'
    
    Returns:
    --------
    pd.DataFrame
        Data with missing values handled
    """
    print("\n" + "="*60)
    print("STEP 1: HANDLING MISSING VALUES")
    print("="*60)
    
    # Check for missing values
    missing_before = df.isnull().sum()
    missing_pct = (missing_before / len(df)) * 100
    
    print("\n📊 Missing Values Before:")
    for col in df.columns:
        if missing_before[col] > 0:
            print(f"   {col}: {missing_before[col]} ({missing_pct[col]:.2f}%)")
    
    if missing_before.sum() == 0:
        print("   ✅ No missing values found!")
        return df
    
    # Apply selected method
    df_clean = df.copy()
    
    if method == 'forward_fill':
        print("\n🔧 Applying forward fill...")
        df_clean = df_clean.fillna(method='ffill')
        # Backward fill any remaining NaN at start
        df_clean = df_clean.fillna(method='bfill')
        
    elif method == 'interpolate':
        print("\n🔧 Applying linear interpolation...")
        df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        
    elif method == 'hybrid':
        print("\n🔧 Applying hybrid approach...")
        # Interpolate for small gaps (<5 values)
        df_clean = df_clean.interpolate(method='linear', limit=5)
        # Forward fill for larger gaps
        df_clean = df_clean.fillna(method='ffill')
        df_clean = df_clean.fillna(method='bfill')
    
    # Verify
    missing_after = df_clean.isnull().sum()
    print("\n📊 Missing Values After:")
    if missing_after.sum() == 0:
        print("   ✅ All missing values handled!")
    else:
        print(f"   ⚠️ Remaining missing values: {missing_after.sum()}")
    
    return df_clean

# ============================================
# STEP 2: TEST STATIONARITY & APPLY DIFFERENCING
# ============================================

def test_stationarity(series, name='Series'):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    """
    result = adfuller(series.dropna())
    
    print(f"\n📊 ADF Test Results for {name}:")
    print(f"   ADF Statistic: {result[0]:.6f}")
    print(f"   p-value: {result[1]:.6f}")
    print(f"   Critical Values:")
    for key, value in result[4].items():
        print(f"      {key}: {value:.3f}")
    
    if result[1] < 0.05:
        print(f"   ✅ STATIONARY (p < 0.05)")
        return True
    else:
        print(f"   ❌ NON-STATIONARY (p >= 0.05)")
        return False

def make_stationary(df, columns=None):
    """
    Apply differencing to non-stationary series
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original time series data
    columns : list
        Columns to check and difference (None = all)
    
    Returns:
    --------
    tuple : (df_stationary, difference_orders)
        Stationary data and dict of difference orders applied
    """
    print("\n" + "="*60)
    print("STEP 2: TESTING STATIONARITY & DIFFERENCING")
    print("="*60)
    
    if columns is None:
        columns = df.columns
    
    df_stationary = df.copy()
    difference_orders = {}
    
    for col in columns:
        print(f"\n🔬 Testing: {col}")
        
        series = df[col].dropna()
        is_stationary = test_stationarity(series, col)
        
        if is_stationary:
            difference_orders[col] = 0
            continue
        
        # Try first difference
        print(f"\n   Applying first difference...")
        diff1 = series.diff().dropna()
        is_stationary_diff1 = test_stationarity(diff1, f"{col} (1st diff)")
        
        if is_stationary_diff1:
            df_stationary[col] = df[col].diff()
            difference_orders[col] = 1
        else:
            # Try second difference
            print(f"\n   Applying second difference...")
            diff2 = diff1.diff().dropna()
            is_stationary_diff2 = test_stationarity(diff2, f"{col} (2nd diff)")
            
            if is_stationary_diff2:
                df_stationary[col] = df[col].diff().diff()
                difference_orders[col] = 2
            else:
                print(f"   ⚠️ Still non-stationary after 2nd difference")
                df_stationary[col] = df[col].diff().diff()
                difference_orders[col] = 2
    
    # Remove NaN created by differencing
    df_stationary = df_stationary.dropna()
    
    print("\n📊 Summary:")
    for col, order in difference_orders.items():
        print(f"   {col}: {'No differencing' if order == 0 else f'{order} difference(s)'}")
    
    return df_stationary, difference_orders

# ============================================
# STEP 3: FEATURE ENGINEERING
# ============================================

def create_features(df, target_col, lags=[1, 3, 6, 12], rolling_windows=[3, 6, 12]):
    """
    Create lag features and rolling statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series data
    target_col : str
        Target variable column name
    lags : list
        Lag periods to create
    rolling_windows : list
        Rolling window sizes for statistics
    
    Returns:
    --------
    pd.DataFrame
        Data with engineered features
    """
    print("\n" + "="*60)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*60)
    
    df_features = df.copy()
    
    # 1. LAG FEATURES
    print(f"\n🔧 Creating lag features for {target_col}...")
    for lag in lags:
        df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
        print(f"   ✓ Created lag_{lag}")
    
    # 2. ROLLING STATISTICS
    print(f"\n🔧 Creating rolling statistics...")
    for window in rolling_windows:
        # Rolling mean
        df_features[f'{target_col}_rolling_mean_{window}'] = \
            df_features[target_col].rolling(window=window).mean()
        
        # Rolling std
        df_features[f'{target_col}_rolling_std_{window}'] = \
            df_features[target_col].rolling(window=window).std()
        
        # Rolling min/max
        df_features[f'{target_col}_rolling_min_{window}'] = \
            df_features[target_col].rolling(window=window).min()
        
        df_features[f'{target_col}_rolling_max_{window}'] = \
            df_features[target_col].rolling(window=window).max()
        
        print(f"   ✓ Created rolling features (window={window})")
    
    # 3. RATE OF CHANGE
    print(f"\n🔧 Creating rate of change features...")
    df_features[f'{target_col}_pct_change_1'] = df_features[target_col].pct_change(1)
    df_features[f'{target_col}_pct_change_12'] = df_features[target_col].pct_change(12)
    print(f"   ✓ Created percent change features")
    
    # 4. MOMENTUM INDICATORS
    print(f"\n🔧 Creating momentum indicators...")
    df_features[f'{target_col}_momentum'] = \
        df_features[target_col] - df_features[target_col].shift(12)
    print(f"   ✓ Created momentum features")
    
    # 5. DATE/TIME FEATURES
    print(f"\n🔧 Creating temporal features...")
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['year'] = df_features.index.year
    print(f"   ✓ Created temporal features")
    
    # Remove NaN created by feature engineering
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    final_rows = len(df_features)
    
    print(f"\n📊 Feature Engineering Summary:")
    print(f"   Original features: {len(df.columns)}")
    print(f"   Total features: {len(df_features.columns)}")
    print(f"   New features: {len(df_features.columns) - len(df.columns)}")
    print(f"   Rows after cleaning: {final_rows} (removed {initial_rows - final_rows})")
    
    return df_features

# ============================================
# STEP 4: BUILD FORECASTING MODELS
# ============================================

# MODEL 1: ARIMA
# ============================================

def build_arima_model(train_data, test_data, order=(1,1,1)):
    """
    Build and evaluate ARIMA model
    
    Parameters:
    -----------
    train_data : pd.Series
        Training data
    test_data : pd.Series
        Testing data
    order : tuple
        ARIMA order (p, d, q)
    
    Returns:
    --------
    dict : Model results and predictions
    """
    print("\n" + "="*60)
    print("MODEL 1: ARIMA")
    print("="*60)
    
    print(f"\n🔧 Training ARIMA{order}...")
    
    # Fit model
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    
    print(f"✅ Model trained successfully!")
    print(f"\nModel Summary:")
    print(model_fit.summary())
    
    # Make predictions
    print(f"\n📊 Generating predictions...")
    predictions = model_fit.forecast(steps=len(test_data))
    
    # Calculate metrics
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    r2 = r2_score(test_data, predictions)
    
    print(f"\n📈 Performance Metrics:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R² Score: {r2:.4f}")
    
    return {
        'model': model_fit,
        'predictions': predictions,
        'metrics': {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
    }

# MODEL 2: PROPHET
# ============================================

def build_prophet_model(train_data, test_data, seasonality_mode='multiplicative'):
    """
    Build and evaluate Prophet model
    
    Parameters:
    -----------
    train_data : pd.Series
        Training data with datetime index
    test_data : pd.Series
        Testing data with datetime index
    seasonality_mode : str
        'additive' or 'multiplicative'
    
    Returns:
    --------
    dict : Model results and predictions
    """
    print("\n" + "="*60)
    print("MODEL 2: PROPHET")
    print("="*60)
    
    print(f"\n🔧 Training Prophet model (seasonality: {seasonality_mode})...")
    
    # Prepare data for Prophet
    train_df = pd.DataFrame({
        'ds': train_data.index,
        'y': train_data.values
    })
    
    # Initialize and fit model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(train_df)
    print(f"✅ Model trained successfully!")
    
    # Create future dataframe
    future = pd.DataFrame({'ds': test_data.index})
    
    # Make predictions
    print(f"\n📊 Generating predictions...")
    forecast = model.predict(future)
    predictions = forecast['yhat'].values
    
    # Calculate metrics
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    r2 = r2_score(test_data, predictions)
    
    print(f"\n📈 Performance Metrics:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R² Score: {r2:.4f}")
    
    return {
        'model': model,
        'forecast': forecast,
        'predictions': predictions,
        'metrics': {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
    }

# MODEL 3: LSTM
# ============================================

def prepare_lstm_data(data, look_back=12):
    """
    Prepare data for LSTM model
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def build_lstm_model(train_data, test_data, look_back=12, epochs=50, batch_size=32):
    """
    Build and evaluate LSTM model
    
    Parameters:
    -----------
    train_data : pd.Series
        Training data
    test_data : pd.Series
        Testing data
    look_back : int
        Number of previous time steps to use
    epochs : int
        Training epochs
    batch_size : int
        Batch size for training
    
    Returns:
    --------
    dict : Model results and predictions
    """
    print("\n" + "="*60)
    print("MODEL 3: LSTM")
    print("="*60)
    
    print(f"\n🔧 Preparing data for LSTM (look_back={look_back})...")
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_data.values.reshape(-1, 1))
    
    # Prepare sequences
    X_train, y_train = prepare_lstm_data(train_scaled.flatten(), look_back)
    X_test, y_test = prepare_lstm_data(test_scaled.flatten(), look_back)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Build LSTM model
    print(f"\n🔧 Building LSTM architecture...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\n🔧 Training LSTM model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )
    
    print(f"✅ Model trained successfully!")
    print(f"   Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"   Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
    # Make predictions
    print(f"\n📊 Generating predictions...")
    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    
    # Get actual test values (accounting for look_back)
    test_actual = test_data.values[look_back:]
    
    # Calculate metrics
    mse = mean_squared_error(test_actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_actual, predictions)
    mape = np.mean(np.abs((test_actual - predictions.flatten()) / test_actual)) * 100
    r2 = r2_score(test_actual, predictions)
    
    print(f"\n📈 Performance Metrics:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R² Score: {r2:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': predictions.flatten(),
        'history': history,
        'look_back': look_back,
        'test_actual': test_actual,
        'metrics': {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
    }

# ============================================
# STEP 5: MODEL EVALUATION & COMPARISON
# ============================================

def compare_models(models_dict, test_data, model_names=['ARIMA', 'Prophet', 'LSTM']):
    """
    Compare performance of multiple models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model results
    test_data : pd.Series
        Test data for comparison
    model_names : list
        Names of models to compare
    
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    print("\n" + "="*60)
    print("STEP 5: MODEL COMPARISON & EVALUATION")
    print("="*60)
    
    comparison_data = []
    
    for name in model_names:
        if name in models_dict:
            metrics = models_dict[name]['metrics']
            comparison_data.append({
                'Model': name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': f"{metrics['MAPE']:.2f}%",
                'R² Score': metrics['R2']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n📊 Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
    best_model_r2 = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
    
    print(f"\n🏆 Best Models:")
    print(f"   Lowest RMSE: {best_model_rmse}")
    print(f"   Highest R²: {best_model_r2}")
    
    return comparison_df

def plot_predictions(test_data, models_dict, model_names=['ARIMA', 'Prophet', 'LSTM']):
    """
    Plot actual vs predicted values for all models
    """
    print("\n📊 Generating comparison plots...")
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=test_data.index[:len(test_data)],
        y=test_data.values,
        mode='lines',
        name='Actual',
        line=dict(color='black', width=3)
    ))
    
    # Model predictions
    colors = ['blue', 'red', 'green']
    for idx, name in enumerate(model_names):
        if name in models_dict:
            predictions = models_dict[name]['predictions']
            
            # Align predictions with test data index
            if name == 'LSTM':
                # LSTM predictions start after look_back period
                look_back = models_dict[name]['look_back']
                x_vals = test_data.index[look_back:look_back+len(predictions)]
            else:
                x_vals = test_data.index[:len(predictions)]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=predictions,
                mode='lines',
                name=f'{name} (MAPE: {models_dict[name]["metrics"]["MAPE"]:.2f}%)',
                line=dict(color=colors[idx], width=2, dash='dash')
            ))
    
    fig.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        height=600,
        template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )
    
    fig.show()
    
    print("✅ Comparison plot generated!")

def plot_residuals(test_data, models_dict, model_names=['ARIMA', 'Prophet', 'LSTM']):
    """
    Plot residuals for all models
    """
    print("\n📊 Generating residual plots...")
    
    fig, axes = plt.subplots(len(model_names), 2, figsize=(14, 4*len(model_names)))
    
    for idx, name in enumerate(model_names):
        if name in models_dict:
            predictions = models_dict[name]['predictions']
            
            # Calculate residuals
            if name == 'LSTM':
                look_back = models_dict[name]['look_back']
                actual = test_data.values[look_back:look_back+len(predictions)]
            else:
                actual = test_data.values[:len(predictions)]
            
            residuals = actual - predictions
            
            # Residual plot
            axes[idx, 0].plot(residuals, linewidth=1.5, color='blue')
            axes[idx, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[idx, 0].set_title(f'{name} - Residuals Over Time')
            axes[idx, 0].set_xlabel('Time')
            axes[idx, 0].set_ylabel('Residual')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Residual histogram
            axes[idx, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[idx, 1].set_title(f'{name} - Residual Distribution')
            axes[idx, 1].set_xlabel('Residual')
            axes[idx, 1].set_ylabel('Frequency')
            axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Residual plots generated!")

# ============================================
# COMPLETE WORKFLOW FUNCTION
# ============================================

def complete_forecasting_workflow(data, target_col, test_size=0.2):
    """
    Run complete forecasting workflow from data processing to evaluation
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with datetime index
    target_col : str
        Target variable to forecast
    test_size : float
        Proportion of data for testing
    
    Returns:
    --------
    dict : All results including processed data and model predictions
    """
    print("\n" + "="*60)
    print("🚀 COMPLETE ECONOMIC FORECASTING WORKFLOW")
    print("="*60)
    print(f"\nTarget Variable: {target_col}")
    print(f"Data Shape: {data.shape}")
    print(f"Date Range: {data.index.min().date()} to {data.index.max().date()}")
    
    # Step 1: Handle missing values
    data_clean = handle_missing_values(data, method='hybrid')
    
    # Step 2: Test stationarity (keep original for modeling)
    target_series = data_clean[target_col]
    
    # Step 3: Feature engineering (optional - for future use)
    # data_features = create_features(data_clean, target_col)
    
    # Split data
    split_idx = int(len(target_series) * (1 - test_size))
    train_data = target_series[:split_idx]
    test_data = target_series[split_idx:]
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(train_data)} samples ({train_data.index.min().date()} to {train_data.index.max().date()})")
    print(f"   Testing: {len(test_data)} samples ({test_data.index.min().date()} to {test_data.index.max().date()})")
    
    # Step 4: Build models
    models_results = {}
    
    # ARIMA
    try:
        models_results['ARIMA'] = build_arima_model(train_data, test_data, order=(2,1,2))
    except Exception as e:
        print(f"\n❌ ARIMA failed: {e}")
    
    # Prophet
    try:
        models_results['Prophet'] = build_prophet_model(train_data, test_data)
    except Exception as e:
        print(f"\n❌ Prophet failed: {e}")
    
    # LSTM
    try:
        models_results['LSTM'] = build_lstm_model(train_data, test_data, look_back=12, epochs=50)
    except Exception as e:
        print(f"\n❌ LSTM failed: {e}")
    
    # Step 5: Compare and evaluate
    comparison_df = compare_models(models_results, test_data)
    
    # Generate plots
    plot_predictions(test_data, models_results)
    plot_residuals(test_data, models_results)
    
    print("\n" + "="*60)
    print("✅ WORKFLOW COMPLETE!")
    print("="*60)
    
    return {
        'data_clean': data_clean,
        'train_data': train_data,
        'test_data': test_data,
        'models': models_results,
        'comparison': comparison_df
    }

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXAMPLE: Running Complete Workflow")
    print("="*60)
    
    # Load your data (replace with your actual data)
    # Assuming you have 'economic_data' from previous setup
    
    # For demo, let's create sample data
    # In practice, use: economic_data = pd.read_csv('economic_indicators.csv', index_col=0, parse_dates=True)
    
    print("\n💡 To run this workflow with your data:")
    print("\n# Load your data")
    print("economic_data = pd.read_csv('economic_indicators.csv', index_col=0, parse_dates=True)")
    print("\n# Run complete workflow")
    print("results = complete_forecasting_workflow(")
    print("    data=economic_data,")
    print("    target_col='Unemployment Rate',  # or 'Consumer Price Index', etc.")
    print("    test_size=0.2")
    print(")")
    print("\n# Access results")
    print("comparison_table = results['comparison']")
    print("best_model = results['models']['Prophet']  # or 'ARIMA', 'LSTM'")
    
    print("\n✅ All functions defined and ready to use!")

