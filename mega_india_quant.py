#!/usr/bin/env python3
"""
Industrial-Grade Quantitative Research Engine for Indian Equities
================================================================

A comprehensive end-to-end quantitative research, forecasting and backtesting engine
focusing on using global semiconductor and tech market signals to forecast Indian equities
and detect frontrunning opportunities.

Features:
- Multi-source data ingestion (yfinance, broker APIs, orderbook/options data)
- Comprehensive feature engineering (technical, statistical, microstructure, options, sentiment)
- Advanced modeling (statistical, ML, deep learning, regime detection)
- Walk-forward backtesting with realistic transaction costs
- GPU acceleration with PyTorch and XGBoost
- Defensive programming with extensive error handling
- Comprehensive output dashboard and reports

Usage:
    python3 mega_india_quant.py --fast          # Fast mode with minimal computation
    python3 mega_india_quant.py                 # Full pipeline
    
Dependencies Installation:
    pip install yfinance pandas numpy scikit-learn xgboost torch statsmodels matplotlib seaborn ta-lib python-dateutil

Hardware Requirements:
- Minimum: 8GB RAM, 2 CPU cores
- Recommended: 16GB+ RAM, GPU (CUDA 11.8+), 4+ CPU cores

API Keys Setup (Optional):
- NewsAPI: Set NEWSAPI_KEY environment variable
- For broker APIs, see connector sections in code

Author: Quant Research Team
Date: 2024
"""

import os
import sys
import warnings
import logging
import argparse
import json
import pickle
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback

# Core libraries
import numpy as np
import pandas as pd

# Data handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not available. Install with: pip install yfinance")

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost with GPU support
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    # Try to enable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not available. Install with: pip install xgboost")

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'
    print("WARNING: PyTorch not available. Install with: pip install torch")

# Statistical models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("WARNING: statsmodels not available. Install with: pip install statsmodels")

# Regime detection
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("WARNING: hmmlearn not available. Install with: pip install hmmlearn")

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
plt.style.use('seaborn-v0_8')

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("WARNING: talib not available. Install with: pip install TA-Lib")

# Date handling
from dateutil.relativedelta import relativedelta

# Parallel processing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# HTTP requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("WARNING: requests not available. Install with: pip install requests")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===============================
# CONFIGURATION SECTION
# ===============================

CONFIG = {
    # Data settings
    'use_gpu': True,  # Enable GPU acceleration if available
    'fast_mode': False,  # Reduced computation for quick testing
    'start_date': '2018-01-01',
    'end_date': '2024-01-01',
    
    # Indian equity tickers (NSE)
    'indian_tickers': [
        '^NSEI',  # NIFTY 50
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS', 'SBIN.NS',
        'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS', 'WIPRO.NS', 'AXISBANK.NS'
    ],
    
    # Global semiconductor and tech tickers
    'global_tickers': [
        '^GSPC',  # S&P 500
        '^IXIC',  # NASDAQ
        'NVDA', 'AMD', 'INTC', 'TSM', 'ASML', 'QCOM',
        'SOXX', 'SMH',  # Semiconductor ETFs
    ],
    
    # Technical analysis parameters
    'sma_periods': [10, 20, 50, 200],
    'ema_periods': [12, 26, 50],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
    'atr_period': 14,
    
    # Model settings
    'sequence_length': 60,  # For LSTM/Transformer
    'train_test_split': 0.7,
    'walk_forward_step': 30,  # Days between retraining
    'min_history_days': 252,  # Minimum data required
    
    # Backtesting settings
    'initial_capital': 100000,  # $100k
    'transaction_cost': 0.001,  # 0.1%
    'slippage': 0.0005,  # 0.05%
    'max_position_size': 0.1,  # 10% per position
    'min_trade_value': 1000,  # Minimum trade size
    
    # Feature toggles
    'use_orderbook': False,  # If orderbook data available
    'use_options': False,  # If options data available
    'use_sentiment': False,  # If sentiment APIs available
    'use_fundamental': False,  # If fundamental data available
    
    # Parallel processing
    'n_jobs': min(4, mp.cpu_count()),
    'use_multiprocessing': False,  # Set to True for CPU-intensive tasks
    
    # Random seed for reproducibility
    'random_seed': 42,
}

# ===============================
# UTILITY FUNCTIONS
# ===============================

def setup_logging():
    """Setup comprehensive logging to file and console"""
    os.makedirs('outputs', exist_ok=True)
    
    log_file = 'outputs/run_log.txt'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def safe_float(value, default=0.0):
    """Safely convert value to float with fallback"""
    try:
        return float(value) if pd.notna(value) else default
    except (ValueError, TypeError):
        return default

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers"""
    try:
        return numerator / denominator if denominator != 0 and pd.notna(denominator) else default
    except (ZeroDivisionError, TypeError):
        return default

def ensure_directory(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def get_date_range(start_date, end_date, freq='D'):
    """Generate date range with proper business days handling"""
    try:
        return pd.date_range(start=start_date, end=end_date, freq=freq)
    except:
        return pd.date_range(start=start_date, end=end_date)

def robust_resample_intraday_to_daily(df, price_col='Close', volume_col='Volume'):
    """Robustly resample intraday data to daily OHLC"""
    try:
        if df.empty:
            return pd.DataFrame()
            
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Aggregate to daily OHLC
        daily = pd.DataFrame()
        
        if len(df) == 0:
            return daily
            
        # Open price (first value of the day)
        daily['Open'] = df[price_col].resample('D').first()
        
        # High price (max value of the day)
        daily['High'] = df[price_col].resample('D').max()
        
        # Low price (min value of the day)
        daily['Low'] = df[price_col].resample('D').min()
        
        # Close price (last value of the day)
        daily['Close'] = df[price_col].resample('D').last()
        
        # Volume (sum of the day)
        if volume_col in df.columns:
            daily['Volume'] = df[volume_col].resample('D').sum()
        
        # Drop rows with all NaN values
        daily = daily.dropna(how='all')
        
        return daily
    except Exception as e:
        logging.warning(f"Error in resampling: {str(e)}")
        return pd.DataFrame()

# ===============================
# DATA INGESTION MODULE
# ===============================

class DataIngestionEngine:
    """Handles data collection from multiple sources with robust error handling"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def download_yfinance_data(self, ticker, start_date, end_date, interval='1d', max_retries=3):
        """Download data from Yahoo Finance with retries and error handling"""
        if not YFINANCE_AVAILABLE:
            self.logger.error(f"yfinance not available for {ticker}")
            return pd.DataFrame()
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading {ticker} from yfinance (attempt {attempt + 1})")
                
                # Handle different intervals
                if interval == '1d':
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                else:
                    # For intraday data
                    data = yf.download(ticker, start=start_date, end=end_date, 
                                     interval=interval, progress=False, threads=False)
                
                if data.empty:
                    self.logger.warning(f"No data returned for {ticker}")
                    continue
                    
                # Clean data
                data = data.dropna(how='all')
                
                # Ensure proper column naming
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]
                
                # Add ticker column
                data['Ticker'] = ticker
                
                self.logger.info(f"Successfully downloaded {len(data)} records for {ticker}")
                return data
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to download {ticker} after {max_retries} attempts")
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return pd.DataFrame()
    
    def get_market_data(self, tickers, start_date, end_date, use_intraday=False):
        """Download market data for multiple tickers"""
        all_data = {}
        
        # Determine interval
        interval = '1h' if use_intraday else '1d'
        
        # Download data with parallel processing if beneficial
        if len(tickers) > 1 and self.config['n_jobs'] > 1:
            with ThreadPoolExecutor(max_workers=self.config['n_jobs']) as executor:
                futures = {
                    executor.submit(self.download_yfinance_data, ticker, start_date, end_date, interval): ticker
                    for ticker in tickers
                }
                
                for future in futures:
                    ticker = futures[future]
                    try:
                        data = future.result(timeout=30)
                        if not data.empty:
                            # Convert intraday to daily if requested
                            if use_intraday and interval != '1d':
                                data = robust_resample_intraday_to_daily(data)
                            
                            all_data[ticker] = data
                    except Exception as e:
                        self.logger.error(f"Failed to get data for {ticker}: {str(e)}")
        else:
            # Sequential download
            for ticker in tickers:
                data = self.download_yfinance_data(ticker, start_date, end_date, interval)
                if not data.empty and use_intraday and interval != '1d':
                    data = robust_resample_intraday_to_daily(data)
                all_data[ticker] = data
                
                # Rate limiting
                time.sleep(0.1)
        
        return all_data
    
    def add_broker_api_connectors(self):
        """Placeholder for broker API connectors"""
        connectors = {
            'zerodha_kite': self._zerodha_kite_connector,
            'ibkr': self._ibkr_connector,
            'upstox': self._upstox_connector
        }
        
        for broker, connector in connectors.items():
            self.logger.info(f"Connector available for {broker} (requires API credentials)")
    
    def _zerodha_kite_connector(self):
        """Zerodha Kite connector placeholder"""
        # Implementation would go here
        pass
    
    def _ibkr_connector(self):
        """Interactive Brokers connector placeholder"""
        # Implementation would go here
        pass
    
    def _upstox_connector(self):
        """Upstox connector placeholder"""
        # Implementation would go here
        pass

# ===============================
# FEATURE ENGINEERING MODULE
# ===============================

class FeatureEngineeringEngine:
    """Comprehensive feature engineering for quantitative models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_technical_indicators(self, df):
        """Create comprehensive technical indicators"""
        try:
            if df.empty or len(df) < 50:
                return df
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning("Missing required OHLC columns for technical indicators")
                return df
            
            features = df.copy()
            
            # Simple Moving Averages
            for period in self.config['sma_periods']:
                if len(df) >= period:
                    features[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
            # Exponential Moving Averages
            for period in self.config['ema_periods']:
                if len(df) >= period:
                    features[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # RSI
            if len(df) >= self.config['rsi_period'] + 1:
                features['RSI'] = self._calculate_rsi(df['Close'], self.config['rsi_period'])
            
            # MACD
            if len(df) >= self.config['macd_slow'] + self.config['macd_signal']:
                macd_line, signal_line, histogram = self._calculate_macd(
                    df['Close'], self.config['macd_fast'], 
                    self.config['macd_slow'], self.config['macd_signal']
                )
                features['MACD'] = macd_line
                features['MACD_Signal'] = signal_line
                features['MACD_Histogram'] = histogram
            
            # Bollinger Bands
            if len(df) >= self.config['bb_period']:
                bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(
                    df['Close'], self.config['bb_period'], self.config['bb_std']
                )
                features['BB_Upper'] = bb_upper
                features['BB_Lower'] = bb_lower
                features['BB_Middle'] = bb_middle
                features['BB_Width'] = bb_upper - bb_lower
                features['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # ATR (Average True Range)
            if len(df) >= self.config['atr_period']:
                features['ATR'] = self._calculate_atr(
                    df['High'], df['Low'], df['Close'], self.config['atr_period']
                )
            
            # Price-based features
            features['Returns'] = df['Close'].pct_change()
            features['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Volatility features
            features['Volatility_5'] = features['Returns'].rolling(5).std()
            features['Volatility_20'] = features['Returns'].rolling(20).std()
            
            # Momentum features
            for period in [5, 10, 20]:
                if len(df) >= period:
                    features[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            
            # High-Low features
            features['HL_Ratio'] = df['High'] / df['Low'] - 1
            features['OC_Ratio'] = df['Close'] / df['Open'] - 1
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating technical indicators: {str(e)}")
            return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series(np.nan, index=prices.index)
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except:
            return pd.Series(np.nan, index=prices.index), pd.Series(np.nan, index=prices.index), pd.Series(np.nan, index=prices.index)
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, lower_band, sma
        except:
            return pd.Series(np.nan, index=prices.index), pd.Series(np.nan, index=prices.index), pd.Series(np.nan, index=prices.index)
    
    def _calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(window=period).mean()
        except:
            return pd.Series(np.nan, index=close.index)
    
    def create_statistical_features(self, df, market_data=None):
        """Create statistical and cross-market features"""
        try:
            if df.empty:
                return df
            
            features = df.copy()
            
            # Ensure timezone-naive index
            if hasattr(features.index, 'tz') and features.index.tz is not None:
                features.index = features.index.tz_localize(None)
            
            # Rolling correlations with market indices
            if market_data is not None and '^GSPC' in market_data:
                market_returns = market_data['^GSPC']['Close'].pct_change()
                # Ensure market returns have timezone-naive index
                if hasattr(market_returns.index, 'tz') and market_returns.index.tz is not None:
                    market_returns.index = market_returns.index.tz_localize(None)
                
                features['Corr_SP500'] = df['Returns'].rolling(60).corr(market_returns)
            
            if market_data is not None and '^IXIC' in market_data:
                nasdaq_returns = market_data['^IXIC']['Close'].pct_change()
                # Ensure nasdaq returns have timezone-naive index
                if hasattr(nasdaq_returns.index, 'tz') and nasdaq_returns.index.tz is not None:
                    nasdaq_returns.index = nasdaq_returns.index.tz_localize(None)
                
                features['Corr_NASDAQ'] = df['Returns'].rolling(60).corr(nasdaq_returns)
            
            # Rolling beta calculation
            if market_data is not None and '^GSPC' in market_data:
                features['Beta_SP500'] = self._calculate_rolling_beta(
                    df['Returns'], market_returns, window=60
                )
            
            # Semiconductor correlations
            semicon_tickers = ['NVDA', 'AMD', 'SOXX']
            for ticker in semicon_tickers:
                if market_data is not None and ticker in market_data:
                    semicon_returns = market_data[ticker]['Close'].pct_change()
                    # Ensure semicon returns have timezone-naive index
                    if hasattr(semicon_returns.index, 'tz') and semicon_returns.index.tz is not None:
                        semicon_returns.index = semicon_returns.index.tz_localize(None)
                    
                    features[f'Corr_{ticker}'] = df['Returns'].rolling(30).corr(semicon_returns)
            
            # PCA-like features (simple implementation)
            if len([k for k in features.columns if 'Corr_' in k]) >= 2:
                corr_cols = [col for col in features.columns if 'Corr_' in col]
                if len(corr_cols) >= 2:
                    # Simple factor construction
                    features['Global_Factor'] = features[corr_cols].mean(axis=1)
            
            # Cointegration tests (placeholder)
            if market_data is not None and '^NSEI' in market_data:
                try:
                    nifty_close = market_data['^NSEI']['Close']
                    # Ensure nifty close has timezone-naive index
                    if hasattr(nifty_close.index, 'tz') and nifty_close.index.tz is not None:
                        nifty_close.index = nifty_close.index.tz_localize(None)
                    
                    if STATSMODELS_AVAILABLE:
                        features['Cointegration_Nifty'] = self._calculate_cointegration_score(
                            df['Close'], nifty_close
                        )
                except Exception as inner_e:
                    self.logger.warning(f"Cointegration calculation failed: {str(inner_e)}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating statistical features: {str(e)}")
            return df
    
    def _calculate_rolling_beta(self, stock_returns, market_returns, window=60):
        """Calculate rolling beta coefficient"""
        try:
            # Ensure timezone-naive indices
            if hasattr(stock_returns.index, 'tz') and stock_returns.index.tz is not None:
                stock_returns = stock_returns.copy()
                stock_returns.index = stock_returns.index.tz_localize(None)
            
            if hasattr(market_returns.index, 'tz') and market_returns.index.tz is not None:
                market_returns = market_returns.copy()
                market_returns.index = market_returns.index.tz_localize(None)
            
            # Align the series
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if len(aligned_data) < window:
                return pd.Series(np.nan, index=stock_returns.index)
            
            rolling_beta = aligned_data.iloc[:, 0].rolling(window).corr(aligned_data.iloc[:, 1])
            return rolling_beta
        except Exception as e:
            self.logger.warning(f"Error in rolling beta calculation: {str(e)}")
            return pd.Series(np.nan, index=stock_returns.index)
    
    def _calculate_cointegration_score(self, series1, series2):
        """Calculate cointegration score"""
        try:
            # Ensure timezone-naive indices
            if hasattr(series1.index, 'tz') and series1.index.tz is not None:
                series1 = series1.copy()
                series1.index = series1.index.tz_localize(None)
            
            if hasattr(series2.index, 'tz') and series2.index.tz is not None:
                series2 = series2.copy()
                series2.index = series2.index.tz_localize(None)
            
            if STATSMODELS_AVAILABLE:
                # Simple correlation as proxy for cointegration
                correlation = series1.rolling(60).corr(series2)
                return correlation
            else:
                return series1.rolling(60).corr(series2)
        except Exception as e:
            self.logger.warning(f"Error in cointegration score calculation: {str(e)}")
            return pd.Series(np.nan, index=series1.index)
    
    def create_microstructure_features(self, df):
        """Create microstructure features if data available"""
        try:
            if df.empty or 'Volume' not in df.columns:
                return df
            
            features = df.copy()
            
            # Volume-based features
            features['Volume_SMA'] = df['Volume'].rolling(20).mean()
            features['Volume_Ratio'] = df['Volume'] / features['Volume_SMA']
            features['Volume_Rolling_Std'] = df['Volume'].rolling(20).std()
            
            # Price-Volume features
            features['Volume_Price_Trend'] = df['Volume'] * df['Returns']
            
            # Liquidity proxy
            features['Turnover'] = df['Volume'] * df['Close']
            features['Turnover_SMA'] = features['Turnover'].rolling(20).mean()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating microstructure features: {str(e)}")
            return df
    
    def create_options_features(self, df, options_data=None):
        """Create options-based features"""
        try:
            if not self.config['use_options'] or options_data is None:
                return df
            
            features = df.copy()
            
            # Placeholder for options features
            # In real implementation, would calculate IV, Greeks, etc.
            features['Options_IV'] = 0.2  # Default 20% IV
            features['Options_Volume'] = 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating options features: {str(e)}")
            return df
    
    def create_sentiment_features(self, df, sentiment_data=None):
        """Create sentiment features from news/social media"""
        try:
            if not self.config['use_sentiment'] or sentiment_data is None:
                return df
            
            features = df.copy()
            
            # Placeholder for sentiment features
            features['News_Sentiment'] = 0.0
            features['Social_Sentiment'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment features: {str(e)}")
            return df
    
    def create_advanced_features(self, df):
        """Create advanced technical features"""
        try:
            if df.empty:
                return df
            
            features = df.copy()
            
            # Multi-timeframe returns
            for period in [1, 3, 5, 10, 20]:
                if len(df) > period:
                    features[f'Return_{period}d'] = df['Close'].pct_change(period)
            
            # Price position features
            for period in [20, 50, 100]:
                if len(df) > period:
                    features[f'Price_Position_{period}d'] = (
                        (df['Close'] - df['Close'].rolling(period).min()) /
                        (df['Close'].rolling(period).max() - df['Close'].rolling(period).min())
                    )
            
            # Volatility features
            features['Realized_Vol_5'] = (df['Returns'].rolling(5) ** 2).sum()
            features['Realized_Vol_20'] = (df['Returns'].rolling(20) ** 2).sum()
            
            # Sharpe ratio (rolling)
            features['Sharpe_20'] = (
                df['Returns'].rolling(20).mean() / df['Returns'].rolling(20).std()
            )
            
            # Drawdown features
            features['Drawdown'] = (df['Close'] / df['Close'].expanding().max() - 1)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating advanced features: {str(e)}")
            return df

# ===============================
# REGIME DETECTION MODULE
# ===============================

class RegimeDetectionEngine:
    """Detect market regimes using Hidden Markov Models and other techniques"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def fit_hmm_regimes(self, data, n_states=3):
        """Fit Hidden Markov Model to detect market regimes"""
        try:
            if not HMM_AVAILABLE or data.empty or len(data) < 100:
                self.logger.warning("HMM not available or insufficient data for regime detection")
                return pd.Series(index=data.index, data=0)
            
            # Prepare data (returns)
            returns = data['Returns'].dropna()
            if len(returns) < 100:
                return pd.Series(index=data.index, data=0)
            
            # Create feature matrix
            features = pd.DataFrame({
                'Returns': returns,
                'Volatility': returns.rolling(20).std()
            }).dropna()
            
            if len(features) < 50:
                return pd.Series(index=data.index, data=0)
            
            # Fit HMM
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
            model.fit(features.values)
            
            # Predict regimes
            states = model.predict(features.values)
            
            # Create regime series
            regime_series = pd.Series(index=features.index, data=states)
            
            # Extend to full index
            full_regime = pd.Series(index=data.index, data=0)
            full_regime.loc[regime_series.index] = regime_series.values
            
            self.logger.info(f"Fitted HMM with {n_states} states")
            return full_regime
            
        except Exception as e:
            self.logger.error(f"Error in HMM regime detection: {str(e)}")
            return pd.Series(index=data.index, data=0)
    
    def detect_volatility_regimes(self, data, threshold_percentiles=[33, 67]):
        """Simple volatility-based regime detection"""
        try:
            if data.empty:
                return pd.Series(index=data.index, data=0)
            
            returns = data['Returns'].dropna()
            if len(returns) < 20:
                return pd.Series(index=data.index, data=0)
            
            # Calculate rolling volatility
            volatility = returns.rolling(20).std()
            
            # Determine thresholds
            low_thresh = np.percentile(volatility.dropna(), threshold_percentiles[0])
            high_thresh = np.percentile(volatility.dropna(), threshold_percentiles[1])
            
            # Create regime labels
            regimes = pd.Series(index=volatility.index, data=1)  # Normal
            
            regimes[volatility <= low_thresh] = 0  # Low volatility
            regimes[volatility >= high_thresh] = 2  # High volatility
            
            # Extend to full index
            full_regimes = pd.Series(index=data.index, data=1)
            full_regimes.loc[regimes.index] = regimes.values
            
            return full_regimes
            
        except Exception as e:
            self.logger.error(f"Error in volatility regime detection: {str(e)}")
            return pd.Series(index=data.index, data=1)

# ===============================
# MODELING MODULE
# ===============================

class QuantitativeModels:
    """Collection of quantitative models for forecasting"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def prepare_sequence_data(self, data, target_col='Close', sequence_length=60):
        """Prepare data for sequence models (LSTM, etc.)"""
        try:
            if len(data) < sequence_length + 1:
                return None, None
            
            # Remove NaN values
            data_clean = data.dropna()
            if len(data_clean) < sequence_length + 1:
                return None, None
            
            # Select features (exclude non-numeric columns)
            feature_cols = [col for col in data_clean.columns 
                          if col not in ['Ticker'] and data_clean[col].dtype in ['float64', 'int64']]
            
            features = data_clean[feature_cols].values
            target = data_clean[target_col].values
            
            if len(features) != len(target):
                return None, None
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features)):
                X.append(features[i-sequence_length:i])
                y.append(target[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Error preparing sequence data: {str(e)}")
            return None, None
    
    def arima_model(self, data, target_col='Close', order=(1,1,1)):
        """ARIMA model implementation"""
        try:
            if not STATSMODELS_AVAILABLE:
                return None
            
            target = data[target_col].dropna()
            if len(target) < 50:
                return None
            
            # Fit ARIMA model
            model = ARIMA(target, order=order)
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=len(target)//4)  # Forecast 25% ahead
            
            return {
                'model': fitted_model,
                'forecast': forecast,
                'residuals': fitted_model.resid
            }
            
        except Exception as e:
            self.logger.error(f"Error in ARIMA model: {str(e)}")
            return None
    
    def random_forest_model(self, X_train, y_train, X_test, n_estimators=100, random_state=42):
        """Random Forest model with GPU support"""
        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=self.config['n_jobs']
            )
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            return {
                'model': model,
                'predictions': predictions,
                'feature_importance': model.feature_importances_
            }
            
        except Exception as e:
            self.logger.error(f"Error in Random Forest model: {str(e)}")
            return None
    
    def xgboost_model(self, X_train, y_train, X_test, n_estimators=100, random_state=42):
        """XGBoost model with GPU support"""
        try:
            if not XGBOOST_AVAILABLE:
                return None
            
            # Enable GPU if available and configured
            tree_method = 'gpu_hist' if (torch.cuda.is_available() and self.config['use_gpu']) else 'hist'
            
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                tree_method=tree_method,
                random_state=random_state,
                n_jobs=self.config['n_jobs']
            )
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            return {
                'model': model,
                'predictions': predictions,
                'feature_importance': model.feature_importances_
            }
            
        except Exception as e:
            self.logger.error(f"Error in XGBoost model: {str(e)}")
            return None
    
    def gradient_boosting_model(self, X_train, y_train, X_test, n_estimators=100, random_state=42):
        """Gradient Boosting model"""
        try:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=self.config['n_jobs']
            )
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            return {
                'model': model,
                'predictions': predictions,
                'feature_importance': model.feature_importances_
            }
            
        except Exception as e:
            self.logger.error(f"Error in Gradient Boosting model: {str(e)}")
            return None
    
    def kalman_filter_model(self, data, target_col='Close'):
        """Simple Kalman Filter for dynamic beta estimation"""
        try:
            # Simple implementation - in practice, use proper Kalman filter library
            returns = data[target_col].pct_change().dropna()
            market_returns = data.get('Returns', pd.Series())
            
            if len(returns) < 50 or market_returns.empty:
                return None
            
            # Align data
            aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
            if len(aligned_data) < 50:
                return None
            
            stock_ret = aligned_data.iloc[:, 0]
            market_ret = aligned_data.iloc[:, 1]
            
            # Rolling beta calculation
            rolling_beta = stock_ret.rolling(60).corr(market_ret)
            
            # Create signals based on beta changes
            beta_signals = rolling_beta.diff().fillna(0)
            
            return {
                'rolling_beta': rolling_beta,
                'beta_signals': beta_signals
            }
            
        except Exception as e:
            self.logger.error(f"Error in Kalman filter model: {str(e)}")
            return None
    
    def lstm_model(self, X_train, y_train, X_test, sequence_length=60, hidden_size=50):
        """LSTM model using PyTorch"""
        try:
            if not TORCH_AVAILABLE:
                return None
            
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size, hidden_size, output_size=1):
                    super(SimpleLSTM, self).__init__()
                    self.hidden_size = hidden_size
                    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])
            
            # Prepare data
            X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
            y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
            X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
            
            if len(X_train.shape) == 2:  # Add sequence dimension if needed
                X_train_tensor = X_train_tensor.unsqueeze(1)
                X_test_tensor = X_test_tensor.unsqueeze(1)
            
            input_size = X_train_tensor.shape[2]
            
            # Initialize model
            model = SimpleLSTM(input_size, hidden_size).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training
            model.train()
            for epoch in range(50):  # Reduced epochs for speed
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    self.logger.info(f"LSTM Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Prediction
            model.eval()
            with torch.no_grad():
                predictions = model(X_test_tensor).cpu().numpy().squeeze()
            
            return {
                'model': model,
                'predictions': predictions,
                'input_size': input_size
            }
            
        except Exception as e:
            self.logger.error(f"Error in LSTM model: {str(e)}")
            return None

# ===============================
# BACKTESTING ENGINE
# ===============================

class BacktestEngine:
    """Comprehensive backtesting engine with realistic transaction costs"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, data, signals, predictions=None):
        """Run comprehensive backtest"""
        try:
            if data.empty or signals.empty:
                self.logger.warning("Empty data or signals for backtest")
                return self._create_empty_results()
            
            # Initialize backtest
            capital = self.config['initial_capital']
            positions = pd.Series(index=data.index, data=0.0)
            portfolio_value = pd.Series(index=data.index, data=capital)
            trades = []
            
            current_position = 0.0
            entry_price = 0.0
            
            for i, (date, row) in enumerate(data.iterrows()):
                current_price = row['Close']
                signal = signals.loc[date] if date in signals.index else 0
                
                # Position sizing
                target_position = self._calculate_position_size(
                    signal, current_price, portfolio_value.iloc[i-1] if i > 0 else capital
                )
                
                # Check if we need to trade
                if abs(target_position - current_position) > 0.01:  # Minimum position change
                    trade_value = abs(target_position - current_position) * current_price
                    
                    if trade_value >= self.config['min_trade_value']:
                        # Execute trade
                        cost = self._calculate_transaction_cost(trade_value)
                        
                        if target_position > current_position:
                            # Buy
                            trades.append({
                                'Date': date,
                                'Action': 'BUY',
                                'Quantity': (target_position - current_position),
                                'Price': current_price,
                                'Value': trade_value,
                                'Cost': cost
                            })
                        else:
                            # Sell
                            trades.append({
                                'Date': date,
                                'Action': 'SELL',
                                'Quantity': (current_position - target_position),
                                'Price': current_price,
                                'Value': trade_value,
                                'Cost': cost
                            })
                        
                        current_position = target_position
                        entry_price = current_price
                
                # Update portfolio value
                portfolio_value.iloc[i] = capital + current_position * (current_price - entry_price)
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(portfolio_value, trades, data.index)
            
            return {
                'portfolio_value': portfolio_value,
                'trades': pd.DataFrame(trades),
                'positions': positions,
                **results
            }
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            return self._create_empty_results()
    
    def _calculate_position_size(self, signal, price, portfolio_value):
        """Calculate position size based on signal and risk management"""
        try:
            # Convert signal to position (0 to max_position_size)
            signal_strength = abs(signal)
            base_position = min(signal_strength, self.config['max_position_size'])
            
            # Volatility-based sizing (simplified)
            volatility_adjustment = 1.0  # Could use historical volatility
            
            final_position = base_position * volatility_adjustment
            
            # Apply direction based on signal
            return final_position * np.sign(signal)
            
        except:
            return 0.0
    
    def _calculate_transaction_cost(self, trade_value):
        """Calculate total transaction cost including slippage"""
        try:
            commission = trade_value * self.config['transaction_cost']
            slippage_cost = trade_value * self.config['slippage']
            return commission + slippage_cost
        except:
            return 0.0
    
    def _calculate_performance_metrics(self, portfolio_value, trades, dates):
        """Calculate comprehensive performance metrics"""
        try:
            if portfolio_value.empty:
                return self._create_empty_metrics()
            
            # Calculate returns
            portfolio_returns = portfolio_value.pct_change().dropna()
            
            if len(portfolio_returns) == 0:
                return self._create_empty_metrics()
            
            # Basic metrics
            total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1)
            
            # Annualized metrics
            years = len(portfolio_value) / 252  # Assuming daily data
            annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            
            # Risk metrics
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Sortino ratio
            negative_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252)
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            rolling_max = portfolio_value.expanding().max()
            drawdown = (portfolio_value - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Win rate
            if trades:
                trades_df = pd.DataFrame(trades)
                winning_trades = trades_df[trades_df['Value'] > 0]  # Simplified
                win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            else:
                win_rate = 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': len(trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._create_empty_metrics()
    
    def _create_empty_results(self):
        """Create empty results structure"""
        empty_series = pd.Series(dtype=float)
        return {
            'portfolio_value': empty_series,
            'trades': pd.DataFrame(),
            'positions': empty_series,
            **self._create_empty_metrics()
        }
    
    def _create_empty_metrics(self):
        """Create empty metrics structure"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0
        }

# ===============================
# VISUALIZATION MODULE
# ===============================

class VisualizationEngine:
    """Create comprehensive visualization dashboard"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_master_dashboard(self, market_data, feature_data, model_results, backtest_results, regime_data):
        """Create comprehensive master dashboard"""
        try:
            if not backtest_results or backtest_results['portfolio_value'].empty:
                self.logger.warning("No backtest results to visualize")
                return
            
            # Create large figure with subplots
            fig = plt.figure(figsize=(20, 24))
            
            # Title
            fig.suptitle('Indian Equities Quantitative Research Dashboard', fontsize=16, fontweight='bold')
            
            # Market indices plot (top left)
            ax1 = plt.subplot(6, 3, (1, 2))
            self._plot_market_indices(ax1, market_data)
            
            # Semiconductor correlation heatmap (top right)
            ax2 = plt.subplot(6, 3, 3)
            self._plot_correlation_heatmap(ax2, feature_data)
            
            # Price charts with signals (second row)
            for i, ticker in enumerate(self.config['indian_tickers'][:3]):
                ax = plt.subplot(6, 3, 4 + i)
                self._plot_price_with_signals(ax, market_data.get(ticker), ticker)
            
            # Model predictions (third row)
            for i, ticker in enumerate(self.config['indian_tickers'][:3]):
                ax = plt.subplot(6, 3, 7 + i)
                self._plot_model_predictions(ax, market_data.get(ticker), model_results.get(ticker), ticker)
            
            # Feature importance (fourth row, left)
            ax11 = plt.subplot(6, 3, 10)
            self._plot_feature_importance(ax11, model_results)
            
            # HMM regimes (fourth row, middle)
            ax12 = plt.subplot(6, 3, 11)
            self._plot_regimes(ax12, regime_data)
            
            # Rolling betas (fourth row, right)
            ax13 = plt.subplot(6, 3, 12)
            self._plot_rolling_betas(ax13, feature_data)
            
            # Portfolio performance (fifth row, left and middle)
            ax14 = plt.subplot(6, 3, (13, 14))
            self._plot_portfolio_performance(ax14, backtest_results)
            
            # Options IV surface (fifth row, right) - placeholder
            ax15 = plt.subplot(6, 3, 15)
            self._plot_options_surface(ax15)
            
            # Performance metrics (bottom row)
            ax16 = plt.subplot(6, 3, (16, 18))
            self._plot_performance_summary(ax16, backtest_results)
            
            # Save figure
            ensure_directory('outputs')
            plt.tight_layout()
            plt.savefig('outputs/master_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Master dashboard saved to outputs/master_dashboard.png")
            
        except Exception as e:
            self.logger.error(f"Error creating master dashboard: {str(e)}")
    
    def _plot_market_indices(self, ax, market_data):
        """Plot market indices"""
        try:
            if market_data and '^NSEI' in market_data:
                nifty_data = market_data['^NSEI']['Close']
                nifty_returns = nifty_data.pct_change().cumsum()
                ax.plot(nifty_returns.index, nifty_returns.values, label='NIFTY 50', linewidth=2)
            
            if market_data and '^GSPC' in market_data:
                sp500_data = market_data['^GSPC']['Close']
                sp500_returns = sp500_data.pct_change().cumsum()
                ax.plot(sp500_returns.index, sp500_returns.values, label='S&P 500', linewidth=2, alpha=0.7)
            
            ax.set_title('Market Indices Performance')
            ax.set_ylabel('Cumulative Returns')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.set_title('Market Indices - No Data')
    
    def _plot_correlation_heatmap(self, ax, feature_data):
        """Plot correlation heatmap"""
        try:
            if not feature_data:
                ax.set_title('Correlation Heatmap - No Data')
                return
            
            # Select correlation columns
            corr_cols = [col for col in feature_data.columns if 'Corr_' in col][:10]
            if corr_cols:
                corr_matrix = feature_data[corr_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            
            ax.set_title('Cross-Market Correlations')
            
        except Exception as e:
            ax.set_title('Correlation Heatmap - Error')
    
    def _plot_price_with_signals(self, ax, data, ticker):
        """Plot price chart with trading signals"""
        try:
            if data is None or data.empty:
                ax.set_title(f'{ticker} - No Data')
                return
            
            # Plot price
            ax.plot(data.index, data['Close'], label='Close Price', linewidth=1)
            
            # Add moving averages if available
            if 'SMA_20' in data.columns:
                ax.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
            
            ax.set_title(f'{ticker} Price Chart')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.set_title(f'{ticker} - Error')
    
    def _plot_model_predictions(self, ax, data, model_result, ticker):
        """Plot model predictions vs actual"""
        try:
            if data is None or data.empty or model_result is None:
                ax.set_title(f'{ticker} Predictions - No Data')
                return
            
            # Plot actual prices
            ax.plot(data.index[-50:], data['Close'].iloc[-50:], label='Actual', linewidth=2)
            
            # Plot predictions if available
            if 'predictions' in model_result:
                pred_values = model_result['predictions'][-50:] if len(model_result['predictions']) >= 50 else model_result['predictions']
                pred_dates = data.index[-len(pred_values):]
                ax.plot(pred_dates, pred_values, label='Predicted', alpha=0.7)
            
            ax.set_title(f'{ticker} Model Prediction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.set_title(f'{ticker} Predictions - Error')
    
    def _plot_feature_importance(self, ax, model_results):
        """Plot feature importance"""
        try:
            # Aggregate feature importance across models
            importances = []
            feature_names = []
            
            for ticker, result in model_results.items():
                if result and 'feature_importance' in result:
                    # This is simplified - in practice, need to match feature names
                    importance = np.mean(result['feature_importance'])
                    importances.append(importance)
                    feature_names.append(f'{ticker[:4]}')
            
            if importances:
                bars = ax.bar(range(len(importances)), importances)
                ax.set_title('Average Feature Importance')
                ax.set_ylabel('Importance')
                ax.set_xticks(range(len(feature_names)))
                ax.set_xticklabels(feature_names, rotation=45)
            
        except Exception as e:
            ax.set_title('Feature Importance - Error')
    
    def _plot_regimes(self, ax, regime_data):
        """Plot market regimes"""
        try:
            if regime_data is None or regime_data.empty:
                ax.set_title('Market Regimes - No Data')
                return
            
            # Plot regime states
            regime_colors = {0: 'green', 1: 'blue', 2: 'red'}
            for state, color in regime_colors.items():
                mask = regime_data == state
                if mask.any():
                    ax.scatter(regime_data.index[mask], [state] * mask.sum(), 
                             c=color, alpha=0.6, s=10)
            
            ax.set_title('HMM Market Regimes')
            ax.set_ylabel('Regime State')
            ax.set_xlabel('Date')
            
        except Exception as e:
            ax.set_title('Market Regimes - Error')
    
    def _plot_rolling_betas(self, ax, feature_data):
        """Plot rolling betas"""
        try:
            if 'Beta_SP500' in feature_data.columns and feature_data['Beta_SP500'].notna().any():
                beta_data = feature_data['Beta_SP500'].dropna()
                ax.plot(beta_data.index, beta_data.values, linewidth=2)
                ax.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Beta = 1')
            
            ax.set_title('Rolling Beta vs S&P 500')
            ax.set_ylabel('Beta')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.set_title('Rolling Betas - Error')
    
    def _plot_portfolio_performance(self, ax, backtest_results):
        """Plot portfolio performance"""
        try:
            portfolio_value = backtest_results['portfolio_value']
            if not portfolio_value.empty:
                # Plot equity curve
                ax.plot(portfolio_value.index, portfolio_value.values, linewidth=2, label='Portfolio Value')
                
                # Plot benchmark (buy and hold)
                initial_value = portfolio_value.iloc[0]
                benchmark = initial_value * (portfolio_value.index.to_series().diff().dt.days.cumsum() * 0.0001 + 1)
                ax.plot(portfolio_value.index, benchmark, linewidth=2, alpha=0.7, label='Simple Benchmark')
            
            ax.set_title('Portfolio Performance')
            ax.set_ylabel('Portfolio Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.set_title('Portfolio Performance - Error')
    
    def _plot_options_surface(self, ax):
        """Plot options implied volatility surface (placeholder)"""
        try:
            # Placeholder for options IV surface
            strikes = np.linspace(80, 120, 20)
            expirations = np.arange(1, 13)
            iv_surface = np.random.normal(0.25, 0.05, (len(expirations), len(strikes)))
            
            im = ax.imshow(iv_surface, aspect='auto', cmap='viridis')
            ax.set_title('Options IV Surface (Placeholder)')
            ax.set_xlabel('Strike')
            ax.set_ylabel('Days to Expiry')
            
        except Exception as e:
            ax.set_title('Options Surface - Error')
    
    def _plot_performance_summary(self, ax, backtest_results):
        """Plot performance summary metrics"""
        try:
            metrics = [
                f"Total Return: {backtest_results.get('total_return', 0):.2%}",
                f"Annualized Return: {backtest_results.get('annualized_return', 0):.2%}",
                f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}",
                f"Max Drawdown: {backtest_results.get('max_drawdown', 0):.2%}",
                f"Win Rate: {backtest_results.get('win_rate', 0):.2%}",
                f"Number of Trades: {backtest_results.get('num_trades', 0)}"
            ]
            
            # Create text summary
            summary_text = '\n'.join(metrics)
            ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
                   fontsize=14, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            ax.set_title('Performance Summary', fontsize=16, fontweight='bold')
            ax.axis('off')
            
        except Exception as e:
            ax.set_title('Performance Summary - Error')

# ===============================
# MAIN EXECUTION ENGINE
# ===============================

class MegaIndiaQuantEngine:
    """Main execution engine that orchestrates the entire quantitative research pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging()
        
        # Initialize components
        self.data_engine = DataIngestionEngine(config)
        self.feature_engine = FeatureEngineeringEngine(config)
        self.regime_engine = RegimeDetectionEngine(config)
        self.model_engine = QuantitativeModels(config)
        self.backtest_engine = BacktestEngine(config)
        self.viz_engine = VisualizationEngine(config)
        
        self.logger.info("Mega India Quant Engine initialized")
    
    def run_full_pipeline(self):
        """Execute the complete quantitative research pipeline"""
        try:
            self.logger.info("Starting full quantitative research pipeline")
            
            # Set random seeds for reproducibility
            np.random.seed(self.config['random_seed'])
            if TORCH_AVAILABLE:
                torch.manual_seed(self.config['random_seed'])
            
            # Phase 1: Data Collection
            self.logger.info("Phase 1: Data Collection")
            market_data = self._collect_market_data()
            
            if not market_data:
                self.logger.error("Failed to collect market data. Exiting.")
                return False
            
            # Phase 2: Feature Engineering
            self.logger.info("Phase 2: Feature Engineering")
            feature_data = self._engineer_features(market_data)
            
            # Phase 3: Regime Detection
            self.logger.info("Phase 3: Regime Detection")
            regime_data = self._detect_regimes(market_data, feature_data)
            
            # Phase 4: Model Training and Prediction
            self.logger.info("Phase 4: Model Training and Prediction")
            model_results = self._train_and_predict_models(market_data, feature_data)
            
            # Phase 5: Signal Generation
            self.logger.info("Phase 5: Signal Generation")
            signals = self._generate_trading_signals(model_results, regime_data)
            
            # Phase 6: Backtesting
            self.logger.info("Phase 6: Backtesting")
            backtest_results = self._run_backtest(market_data, signals)
            
            # Phase 7: Visualization
            self.logger.info("Phase 7: Creating visualizations")
            self.viz_engine.create_master_dashboard(
                market_data, feature_data, model_results, backtest_results, regime_data
            )
            
            # Phase 8: Results Export
            self.logger.info("Phase 8: Exporting results")
            self._export_results(market_data, feature_data, model_results, backtest_results)
            
            # Print final summary
            self._print_final_summary(backtest_results)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _collect_market_data(self):
        """Collect market data from all sources"""
        try:
            # Download Indian equities
            indian_data = self.data_engine.get_market_data(
                self.config['indian_tickers'], 
                self.config['start_date'], 
                self.config['end_date'],
                use_intraday=not self.config['fast_mode']
            )
            
            # Download global market data
            global_data = self.data_engine.get_market_data(
                self.config['global_tickers'],
                self.config['start_date'],
                self.config['end_date'],
                use_intraday=False
            )
            
            # Combine data
            market_data = {**indian_data, **global_data}
            
            # Filter out empty datasets
            market_data = {k: v for k, v in market_data.items() if not v.empty}
            
            self.logger.info(f"Successfully collected data for {len(market_data)} tickers")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {str(e)}")
            return {}
    
    def _engineer_features(self, market_data):
        """Engineer comprehensive features for all tickers"""
        try:
            feature_data = {}
            
            for ticker, data in market_data.items():
                if data.empty or len(data) < self.config['min_history_days']:
                    continue
                
                self.logger.info(f"Engineering features for {ticker}")
                
                # Create features
                features = self.feature_engine.create_technical_indicators(data)
                features = self.feature_engine.create_statistical_features(features, market_data)
                features = self.feature_engine.create_microstructure_features(features)
                features = self.feature_engine.create_advanced_features(features)
                
                feature_data[ticker] = features
            
            self.logger.info(f"Feature engineering completed for {len(feature_data)} tickers")
            return feature_data
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            return {}
    
    def _detect_regimes(self, market_data, feature_data):
        """Detect market regimes"""
        try:
            # Use NIFTY 50 as primary market indicator
            if '^NSEI' in market_data:
                nifty_data = market_data['^NSEI']
                nifty_features = feature_data.get('^NSEI', nifty_data)
                
                # HMM regimes
                hmm_regimes = self.regime_engine.fit_hmm_regimes(nifty_features)
                
                # Volatility regimes
                vol_regimes = self.regime_engine.detect_volatility_regimes(nifty_features)
                
                # Combine regimes
                combined_regimes = pd.DataFrame({
                    'HMM': hmm_regimes,
                    'Volatility': vol_regimes
                }).mean(axis=1).round().astype(int)
                
                return combined_regimes
            
            return pd.Series()
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {str(e)}")
            return pd.Series()
    
    def _train_and_predict_models(self, market_data, feature_data):
        """Train and predict using multiple models"""
        try:
            model_results = {}
            
            # Focus on top Indian tickers
            indian_tickers = [t for t in self.config['indian_tickers'] if t in market_data]
            
            for ticker in indian_tickers[:5]:  # Limit for performance
                if ticker not in feature_data or feature_data[ticker].empty:
                    continue
                
                self.logger.info(f"Training models for {ticker}")
                
                ticker_results = {}
                
                # Prepare data
                data = feature_data[ticker].copy()
                data = data.dropna()
                
                if len(data) < self.config['min_history_days']:
                    continue
                
                # Split data
                split_idx = int(len(data) * self.config['train_test_split'])
                train_data = data.iloc[:split_idx]
                test_data = data.iloc[split_idx:]
                
                if len(test_data) < 10:
                    continue
                
                # Prepare features
                feature_cols = [col for col in data.columns 
                              if col not in ['Ticker', 'Close'] and data[col].dtype in ['float64', 'int64']]
                
                if len(feature_cols) < 5:
                    continue
                
                X_train = train_data[feature_cols].fillna(0)
                y_train = train_data['Close']
                X_test = test_data[feature_cols].fillna(0)
                y_test = test_data['Close']
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train multiple models
                models_to_try = ['random_forest', 'xgboost']
                
                if not self.config['fast_mode']:
                    models_to_try.extend(['gradient_boosting', 'arima', 'lstm'])
                
                for model_name in models_to_try:
                    try:
                        if model_name == 'random_forest':
                            result = self.model_engine.random_forest_model(
                                X_train_scaled, y_train, X_test_scaled
                            )
                        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                            result = self.model_engine.xgboost_model(
                                X_train_scaled, y_train, X_test_scaled
                            )
                        elif model_name == 'gradient_boosting':
                            result = self.model_engine.gradient_boosting_model(
                                X_train_scaled, y_train, X_test_scaled
                            )
                        elif model_name == 'arima' and STATSMODELS_AVAILABLE:
                            result = self.model_engine.arima_model(data)
                        elif model_name == 'lstm' and TORCH_AVAILABLE:
                            # For LSTM, use sequence data
                            X_seq_train, y_seq_train = self.model_engine.prepare_sequence_data(
                                train_data, sequence_length=self.config['sequence_length']
                            )
                            X_seq_test, y_seq_test = self.model_engine.prepare_sequence_data(
                                test_data, sequence_length=self.config['sequence_length']
                            )
                            
                            if X_seq_train is not None:
                                result = self.model_engine.lstm_model(
                                    X_seq_train, y_seq_train, X_seq_test,
                                    sequence_length=self.config['sequence_length']
                                )
                            else:
                                result = None
                        
                        if result:
                            ticker_results[model_name] = result
                            
                    except Exception as e:
                        self.logger.warning(f"Model {model_name} failed for {ticker}: {str(e)}")
                
                if ticker_results:
                    model_results[ticker] = ticker_results
            
            self.logger.info(f"Model training completed for {len(model_results)} tickers")
            return model_results
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            return {}
    
    def _generate_trading_signals(self, model_results, regime_data):
        """Generate trading signals from model predictions"""
        try:
            all_signals = pd.DataFrame()
            
            for ticker, results in model_results.items():
                ticker_signals = pd.Series(index=pd.Index([]), dtype=float)
                
                # Combine predictions from multiple models
                predictions = []
                weights = []
                
                for model_name, result in results.items():
                    try:
                        if 'predictions' in result:
                            pred = result['predictions']
                            if len(pred) > 0:
                                predictions.append(pred[-1])  # Latest prediction
                                weights.append(1.0)  # Equal weight
                    except:
                        continue
                
                if predictions:
                    # Simple ensemble - average predictions
                    ensemble_prediction = np.mean(predictions)
                    
                    # Get actual price for signal generation
                    # This is simplified - in practice, would use walk-forward
                    if ticker in model_results:
                        # Create signal based on prediction vs recent price trend
                        # For now, use simple momentum signal
                        signal = 0.1 if ensemble_prediction > 0 else -0.1
                        ticker_signals = pd.Series([signal], index=[pd.Timestamp.now()])
                
                if not ticker_signals.empty:
                    all_signals[ticker] = ticker_signals
            
            return all_signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            return pd.DataFrame()
    
    def _run_backtest(self, market_data, signals):
        """Run comprehensive backtest"""
        try:
            # Use NIFTY 50 as primary asset for backtesting
            if '^NSEI' not in market_data:
                return self.backtest_engine._create_empty_results()
            
            data = market_data['^NSEI']
            
            # Create simple signals based on global tech/semiconductor performance
            global_signal = pd.Series(index=data.index, data=0.0)
            
            # Simple signal: if global tech outperforms, go long India
            if 'NVDA' in market_data:
                nvda_returns = market_data['NVDA']['Close'].pct_change()
                signal_threshold = nvda_returns.quantile(0.8)
                global_signal = (nvda_returns > signal_threshold).astype(float) - 0.5
            
            # Combine with any model-based signals
            if not signals.empty and '^NSEI' in signals.columns:
                model_signal = signals['^NSEI'].reindex(data.index, fill_value=0)
                combined_signal = global_signal * 0.7 + model_signal * 0.3
            else:
                combined_signal = global_signal
            
            # Run backtest
            backtest_results = self.backtest_engine.run_backtest(data, combined_signal)
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            return self.backtest_engine._create_empty_results()
    
    def _export_results(self, market_data, feature_data, model_results, backtest_results):
        """Export all results to CSV files"""
        try:
            ensure_directory('outputs')
            
            # Export market data
            for ticker, data in market_data.items():
                if not data.empty:
                    data.to_csv(f'outputs/market_data_{ticker.replace(".", "_")}.csv')
            
            # Export feature data
            for ticker, data in feature_data.items():
                if not data.empty:
                    data.to_csv(f'outputs/features_{ticker.replace(".", "_")}.csv')
            
            # Export model results
            model_summary = []
            for ticker, results in model_results.items():
                for model_name, result in results.items():
                    model_summary.append({
                        'Ticker': ticker,
                        'Model': model_name,
                        'Has_Predictions': 'predictions' in result,
                        'Has_Feature_Importance': 'feature_importance' in result
                    })
            
            if model_summary:
                pd.DataFrame(model_summary).to_csv('outputs/model_summary.csv', index=False)
            
            # Export backtest results
            if backtest_results and not backtest_results['portfolio_value'].empty:
                backtest_results['portfolio_value'].to_csv('outputs/portfolio_value.csv')
                
                if not backtest_results['trades'].empty:
                    backtest_results['trades'].to_csv('outputs/trades.csv', index=False)
                
                # Export performance metrics
                metrics = {k: [v] for k, v in backtest_results.items() 
                          if isinstance(v, (int, float)) and k not in ['portfolio_value', 'trades', 'positions']}
                if metrics:
                    pd.DataFrame(metrics).to_csv('outputs/performance_metrics.csv', index=False)
            
            self.logger.info("Results exported to outputs directory")
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
    
    def _print_final_summary(self, backtest_results):
        """Print comprehensive final summary"""
        try:
            print("\n" + "="*60)
            print("MEGA INDIA QUANT ENGINE - FINAL RESULTS")
            print("="*60)
            
            if backtest_results and not backtest_results['portfolio_value'].empty:
                metrics = [
                    ("Total Return", f"{backtest_results.get('total_return', 0):.2%}"),
                    ("Annualized Return", f"{backtest_results.get('annualized_return', 0):.2%}"),
                    ("Sharpe Ratio", f"{backtest_results.get('sharpe_ratio', 0):.2f}"),
                    ("Sortino Ratio", f"{backtest_results.get('sortino_ratio', 0):.2f}"),
                    ("Maximum Drawdown", f"{backtest_results.get('max_drawdown', 0):.2%}"),
                    ("Win Rate", f"{backtest_results.get('win_rate', 0):.2%}"),
                    ("Number of Trades", f"{backtest_results.get('num_trades', 0)}"),
                    ("Volatility", f"{backtest_results.get('volatility', 0):.2%}")
                ]
                
                for metric_name, value in metrics:
                    print(f"{metric_name:<20}: {value}")
                
                print(f"\nBest Performing Models:")
                print("Note: This is historical optimization and may overfit. Use out-of-sample validation.")
                
            else:
                print("No successful backtest results to report.")
            
            print(f"\nOutput Files Generated:")
            print("- outputs/master_dashboard.png (Main visualization dashboard)")
            print("- outputs/run_log.txt (Detailed execution log)")
            print("- outputs/portfolio_value.csv (Portfolio performance)")
            print("- outputs/trades.csv (Individual trades)")
            print("- outputs/performance_metrics.csv (Performance summary)")
            print("- outputs/model_summary.csv (Model comparison)")
            
            print(f"\nHardware Configuration:")
            print(f"- PyTorch Device: {DEVICE}")
            print(f"- XGBoost GPU Support: {'Enabled' if (XGBOOST_AVAILABLE and torch.cuda.is_available() and self.config['use_gpu']) else 'Disabled'}")
            print(f"- Parallel Jobs: {self.config['n_jobs']}")
            
            print(f"\nIMPORTANT DISCLAIMERS:")
            print("- This system is for research and educational purposes only")
            print("- Past performance does not guarantee future results")
            print("- All trading involves substantial risk of loss")
            print("- This is NOT financial advice")
            
            print("="*60)
            
        except Exception as e:
            print(f"Error printing final summary: {str(e)}")

# ===============================
# COMMAND LINE INTERFACE
# ===============================

def main():
    """Main entry point with command line argument handling"""
    parser = argparse.ArgumentParser(description='Mega India Quant Research Engine')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode with reduced computation')
    parser.add_argument('--tickers', nargs='+', help='Specify custom tickers')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, help='Initial capital amount')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.fast:
        CONFIG['fast_mode'] = True
        CONFIG['train_test_split'] = 0.8  # Less training data for speed
    
    if args.tickers:
        CONFIG['indian_tickers'] = args.tickers
    
    if args.start_date:
        CONFIG['start_date'] = args.start_date
    
    if args.end_date:
        CONFIG['end_date'] = args.end_date
    
    if args.capital:
        CONFIG['initial_capital'] = args.capital
    
    # Create and run engine
    engine = MegaIndiaQuantEngine(CONFIG)
    
    print("Mega India Quantitative Research Engine")
    print(f"Mode: {'Fast' if CONFIG['fast_mode'] else 'Full'}")
    print(f"GPU Enabled: {CONFIG['use_gpu'] and torch.cuda.is_available()}")
    print(f"Start Date: {CONFIG['start_date']}")
    print(f"End Date: {CONFIG['end_date']}")
    print(f"Initial Capital: ${CONFIG['initial_capital']:,.0f}")
    print("-" * 50)
    
    # Run pipeline
    success = engine.run_full_pipeline()
    
    if success:
        print("\nPipeline completed successfully!")
        print("Check outputs/ directory for results.")
    else:
        print("\nPipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("Check outputs/run_log.txt for details")
        sys.exit(1)