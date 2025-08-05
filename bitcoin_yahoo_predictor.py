#!/usr/bin/env python3
"""
Bitcoin Price Prediction System - Yahoo Finance Data Source
Alternative data source using Yahoo Finance instead of Binance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class BitcoinYahooPredictor:
    def __init__(self):
        """Initialize Yahoo Finance Bitcoin predictor"""
        self.symbol = "BTC-USD"
        self.scaler = StandardScaler()
        self.models = {}
        self.sequence_length = 60
        
        print("🚀 Bitcoin Yahoo Finance Predictor Initialized")
        print(f"📊 Symbol: {self.symbol}")
        
    def get_yahoo_data(self, period="1y", interval="1d"):
        """Get Bitcoin data from Yahoo Finance"""
        try:
            print(f"📥 Fetching data from Yahoo Finance...")
            print(f"   Period: {period}, Interval: {interval}")
            
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print("❌ No data received from Yahoo Finance")
                return None
                
            # Rename columns to match our system
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            print(f"✅ Data fetched successfully: {len(data)} records")
            print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"💰 Latest price: ${data['close'].iloc[-1]:,.2f}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching Yahoo Finance data: {str(e)}")
            return None
    
    def create_technical_features(self, df):
        """Create advanced technical features"""
        try:
            print("🔧 Creating technical features...")
            
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
            
            # Technical indicators
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std()
            
            # Price position in range
            df['price_position'] = (df['close'] - df['low'].rolling(window=14).min()) / \
                                 (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())
            
            # Lag features
            for lag in [1, 2, 3, 5, 7]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            # Time-based features
            df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
            df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
            df['month'] = df.index.month if hasattr(df.index, 'month') else 1
            
            print(f"✅ Technical features created: {len(df.columns)} total columns")
            return df
            
        except Exception as e:
            print(f"❌ Error creating features: {str(e)}")
            return df
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        try:
            print("📊 Preparing training data...")
            
            # Select feature columns (exclude non-numeric and target)
            feature_columns = [col for col in df.columns if col not in ['close'] and df[col].dtype in ['float64', 'int64']]
            
            # Remove rows with NaN values
            df_clean = df[feature_columns + ['close']].dropna()
            
            if len(df_clean) < self.sequence_length:
                print(f"❌ Insufficient data after cleaning: {len(df_clean)} rows")
                return None, None, None, None
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(df_clean)):
                X.append(df_clean[feature_columns].iloc[i-self.sequence_length:i].values.flatten())
                y.append(df_clean['close'].iloc[i])
            
            X, y = np.array(X), np.array(y)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            print(f"✅ Training data prepared:")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Test samples: {len(X_test)}")
            print(f"   Features per sample: {X_train.shape[1]}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            print(f"❌ Error preparing training data: {str(e)}")
            return None, None, None, None
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train prediction models"""
        try:
            print("🤖 Training prediction models...")
            
            # Random Forest
            print("   Training Random Forest...")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            
            # Linear Regression
            print("   Training Linear Regression...")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_pred = lr_model.predict(X_test)
            lr_mae = mean_absolute_error(y_test, lr_pred)
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
            
            self.models = {
                'Random Forest': {
                    'model': rf_model,
                    'mae': rf_mae,
                    'rmse': rf_rmse,
                    'predictions': rf_pred
                },
                'Linear Regression': {
                    'model': lr_model,
                    'mae': lr_mae,
                    'rmse': lr_rmse,
                    'predictions': lr_pred
                }
            }
            
            print("✅ Models trained successfully:")
            print(f"   Random Forest - MAE: ${rf_mae:,.2f}, RMSE: ${rf_rmse:,.2f}")
            print(f"   Linear Regression - MAE: ${lr_mae:,.2f}, RMSE: ${lr_rmse:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training models: {str(e)}")
            return False
    
    def generate_30day_predictions(self, df):
        """Generate 30-day predictions"""
        try:
            print("🔮 Generating 30-day predictions...")
            
            # Get latest data for prediction
            feature_columns = [col for col in df.columns if col not in ['close'] and df[col].dtype in ['float64', 'int64']]
            latest_data = df[feature_columns].tail(self.sequence_length).values.flatten()
            latest_data_scaled = self.scaler.transform([latest_data])
            
            predictions = {}
            current_date = datetime.now()
            
            for day in range(1, 31):
                pred_date = current_date + timedelta(days=day)
                
                # Get predictions from both models
                rf_pred = self.models['Random Forest']['model'].predict(latest_data_scaled)[0]
                lr_pred = self.models['Linear Regression']['model'].predict(latest_data_scaled)[0]
                
                # Ensemble prediction (weighted average)
                rf_weight = 0.6  # Random Forest typically performs better
                lr_weight = 0.4
                ensemble_pred = (rf_pred * rf_weight) + (lr_pred * lr_weight)
                
                # Calculate confidence (inverse of prediction variance)
                pred_variance = np.var([rf_pred, lr_pred])
                confidence = max(50, min(95, 90 - (pred_variance / 1000)))
                
                predictions[str(day)] = {
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'day_name': pred_date.strftime('%A'),
                    'predicted_price': round(ensemble_pred, 2),
                    'model_predictions': {
                        'Random Forest': round(rf_pred, 2),
                        'Linear Regression': round(lr_pred, 2)
                    },
                    'confidence': round(confidence, 1)
                }
            
            print(f"✅ 30-day predictions generated successfully")
            return predictions
            
        except Exception as e:
            print(f"❌ Error generating predictions: {str(e)}")
            return {}
    
    def save_results(self, predictions):
        """Save prediction results to JSON file"""
        try:
            print("💾 Saving results...")
            
            result_data = {
                'generated_at': datetime.now().isoformat(),
                'data_source': 'Yahoo Finance',
                'symbol': self.symbol,
                'model_performance': {
                    'Random Forest': {
                        'test_mae': self.models['Random Forest']['mae'],
                        'test_rmse': self.models['Random Forest']['rmse']
                    },
                    'Linear Regression': {
                        'test_mae': self.models['Linear Regression']['mae'],
                        'test_rmse': self.models['Linear Regression']['rmse']
                    }
                },
                'predictions': predictions
            }
            
            filename = 'bitcoin_30day_predictions_yahoo.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
            return None

def main():
    """Main function"""
    try:
        print("🚀 Bitcoin Yahoo Finance Prediction System")
        print("=" * 50)
        
        # Initialize predictor
        predictor = BitcoinYahooPredictor()
        
        # Get data from Yahoo Finance
        df = predictor.get_yahoo_data(period="2y", interval="1d")
        if df is None:
            print("❌ Failed to get data from Yahoo Finance")
            return
        
        # Create technical features
        df = predictor.create_technical_features(df)
        
        # Prepare training data
        X_train, X_test, y_train, y_test = predictor.prepare_training_data(df)
        if X_train is None:
            print("❌ Failed to prepare training data")
            return
        
        # Train models
        if not predictor.train_models(X_train, X_test, y_train, y_test):
            print("❌ Failed to train models")
            return
        
        # Generate 30-day predictions
        predictions = predictor.generate_30day_predictions(df)
        if not predictions:
            print("❌ Failed to generate predictions")
            return
        
        # Save results
        filename = predictor.save_results(predictions)
        if filename:
            print(f"\n🎉 Prediction completed successfully!")
            print(f"📁 Results saved to: {filename}")
            print(f"📊 30-day predictions generated using Yahoo Finance data")
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
    
    input("\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()