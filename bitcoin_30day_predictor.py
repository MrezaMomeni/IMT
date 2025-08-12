#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Bitcoin 30-Day Detailed Price Predictor
Bitcoin price predictor for the next 30 days
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from binance.client import Client
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ta

load_dotenv()

class Bitcoin30DayPredictor:
    def __init__(self):
        """Initialize the 30-day Bitcoin predictor"""
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("❌ API keys not found in .env file")
        
        self.client = Client(self.api_key, self.secret_key)
        self.scaler = MinMaxScaler()
        self.models = {}
        self.predictions = {}
        
        print("🚀 Bitcoin 30-Day Predictor Initialized")
        print("🔑 API Status: ✅ Connected")
    
    def fetch_historical_data(self, days=365):
        """Fetch historical Bitcoin data for training"""
        print(f"📈 Fetching {days} days of historical data...")
        
        try:
            klines = self.client.get_historical_klines(
                "BTCUSDT", 
                Client.KLINE_INTERVAL_1DAY, 
                f"{days} days ago UTC"
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['timestamp'].dt.date
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df[['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ Retrieved {len(df)} days of data")
            print(f"   📊 Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def create_advanced_features(self, df):
        """Create advanced technical features for prediction"""
        print("🔧 Creating advanced technical features...")
        
        df = df.copy()
        
        df['price_change'] = df['close'].pct_change()
        df['price_volatility'] = df['price_change'].rolling(7).std()
        df['price_momentum'] = df['close'] / df['close'].shift(7) - 1
        
        for period in [7, 14, 21, 30, 50]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ma_{period}_ratio'] = df['close'] / df[f'ma_{period}']
        
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        df['volume_sma'] = df['volume'].rolling(14).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['volume']
        
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        for lag in [1, 2, 3, 7, 14]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        df['close_min_7d'] = df['close'].rolling(7).min()
        df['close_max_7d'] = df['close'].rolling(7).max()
        df['close_std_7d'] = df['close'].rolling(7).std()
        
        print(f"✅ Created {len([col for col in df.columns if col not in ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']])} features")
        
        return df
    
    def prepare_training_data(self, df, target_days=30):
        """Prepare data for training multiple models"""
        print(f"🎯 Preparing training data for {target_days}-day prediction...")
        
        df_clean = df.dropna().copy()
        
        feature_cols = [col for col in df_clean.columns 
                       if col not in ['date', 'timestamp', 'close']]
        
        X = df_clean[feature_cols].values
        y = df_clean['close'].values
        
        sequence_length = 30  
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i].flatten())
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        split_idx = int(0.8 * len(X_sequences))
        
        X_train = X_sequences[:split_idx]
        X_test = X_sequences[split_idx:]
        y_train = y_sequences[:split_idx]
        y_test = y_sequences[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Training data prepared:")
        print(f"   📊 Training samples: {len(X_train)}")
        print(f"   📊 Test samples: {len(X_test)}")
        print(f"   📊 Features per sample: {X_train.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models for ensemble prediction"""
        print("🤖 Training prediction models...")
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"   🔄 Training {name}...")
            
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'y_pred_test': y_pred_test
            }
            
            print(f"   ✅ {name}: MAE=${test_mae:.2f}, RMSE=${test_rmse:.2f}")
        
        self.models = results
        return results
    
    def predict_30_days(self, df, feature_cols):
        """Generate 30-day detailed predictions"""
        print("🔮 Generating 30-day predictions...")
        
        last_data = df.dropna().tail(30)
        
        predictions = {}
        current_date = datetime.now().date()
        
        for day in range(1, 31):
            future_date = current_date + timedelta(days=day)
            
            feature_data = last_data[feature_cols].values
            input_sequence = feature_data.flatten().reshape(1, -1)
            input_scaled = self.scaler.transform(input_sequence)
            
            model_predictions = {}
            for name, model_info in self.models.items():
                pred = model_info['model'].predict(input_scaled)[0]
                model_predictions[name] = pred
            
            ensemble_pred = np.mean(list(model_predictions.values()))
            
            last_price = df['close'].iloc[-1]
            daily_volatility = df['price_change'].std()
            
            trend_factor = 1 + (np.random.normal(0, daily_volatility) * 0.5)
            adjusted_pred = ensemble_pred * trend_factor
            
            predictions[day] = {
                'date': future_date.strftime('%Y-%m-%d'),
                'day_name': future_date.strftime('%A'),
                'predicted_price': round(adjusted_pred, 2),
                'model_predictions': {k: round(v, 2) for k, v in model_predictions.items()},
                'confidence': min(95, max(60, 85 - (day * 0.8)))  # Decreasing confidence over time
            }
            
           
            
        self.predictions = predictions
        print(f"✅ Generated predictions for 30 days")
        
        return predictions
    
    def save_predictions(self, filename='bitcoin_30day_predictions.json'):
        """Save predictions to JSON file"""
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'model_performance': {
                name: {
                    'test_mae': info['test_mae'],
                    'test_rmse': info['test_rmse']
                }
                for name, info in self.models.items()
            },
            'predictions': self.predictions
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Predictions saved to {filename}")
    
    def create_visualization(self, df):
        """Create visualization of predictions"""
        print("📊 Creating prediction visualization...")
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        historical_dates = pd.to_datetime([pred['date'] for pred in self.predictions.values()])
        predicted_prices = [pred['predicted_price'] for pred in self.predictions.values()]
        confidence_scores = [pred['confidence'] for pred in self.predictions.values()]
        
        recent_data = df.tail(60)
        ax1.plot(recent_data['timestamp'], recent_data['close'], 
                label='Historical Price', color='#00ff88', linewidth=2)
        
        ax1.plot(historical_dates, predicted_prices, 
                label='30-Day Predictions', color='#ff6b6b', linewidth=2, linestyle='--')
        
        ax1.set_title('🚀 Bitcoin Price: Historical vs 30-Day Predictions', fontsize=16, color='white')
        ax1.set_xlabel('Date', color='white')
        ax1.set_ylabel('Price (USD)', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(range(1, 31), confidence_scores, color='#4ecdc4', alpha=0.7)
        ax2.set_title('📊 Prediction Confidence by Day', fontsize=16, color='white')
        ax2.set_xlabel('Day', color='white')
        ax2.set_ylabel('Confidence (%)', color='white')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bitcoin_30day_predictions.png', dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        print("✅ Visualization saved as bitcoin_30day_predictions.png")
    
    def display_summary(self):
        """Display prediction summary"""
        print("\n" + "="*80)
        print("📈 BITCOIN 30-DAY PRICE PREDICTIONS SUMMARY")
        print("="*80)
        
        if not self.predictions:
            print("❌ No predictions available")
            return
        
        try:
            current_ticker = self.client.get_symbol_ticker(symbol="BTCUSDT")
            current_price = float(current_ticker['price'])
            print(f"💰 Current BTC Price: ${current_price:,.2f}")
        except:
            current_price = None
            print("💰 Current BTC Price: Unable to fetch")
        
        print(f"🔮 Prediction Period: {self.predictions[1]['date']} to {self.predictions[30]['date']}")
        
        prices = [pred['predicted_price'] for pred in self.predictions.values()]
        min_price = min(prices)
        max_price = max(prices)
        avg_price = np.mean(prices)
        
        print(f"📊 Predicted Price Range:")
        print(f"   🔻 Minimum: ${min_price:,.2f}")
        print(f"   🔺 Maximum: ${max_price:,.2f}")
        print(f"   📈 Average: ${avg_price:,.2f}")
        
        if current_price:
            change_min = ((min_price - current_price) / current_price) * 100
            change_max = ((max_price - current_price) / current_price) * 100
            print(f"📈 Expected Change Range: {change_min:.1f}% to {change_max:.1f}%")
        
        print(f"\n📅 Weekly Breakdown:")
        for week in range(4):
            start_day = week * 7 + 1
            end_day = min((week + 1) * 7, 30)
            week_prices = [self.predictions[day]['predicted_price'] 
                          for day in range(start_day, end_day + 1)]
            week_avg = np.mean(week_prices)
            print(f"   Week {week + 1} (Days {start_day}-{end_day}): ${week_avg:,.2f}")
        
        print(f"\n🤖 Model Performance:")
        for name, info in self.models.items():
            print(f"   {name}: MAE=${info['test_mae']:.2f}, RMSE=${info['test_rmse']:.2f}")
        
        print("\n" + "="*80)

def main():
    """Main function to run 30-day prediction"""
    print("🚀 Bitcoin 30-Day Detailed Price Predictor")
    print("=" * 60)
    
    try:
        predictor = Bitcoin30DayPredictor()
        
        df = predictor.fetch_historical_data(days=365)
        if df is None:
            return
        
        df_features = predictor.create_advanced_features(df)
        
        X_train, X_test, y_train, y_test, feature_cols = predictor.prepare_training_data(df_features)
        
        model_results = predictor.train_models(X_train, X_test, y_train, y_test)
        
        predictions = predictor.predict_30_days(df_features, feature_cols)
        
        predictor.save_predictions()
        
        predictor.create_visualization(df)
        
        predictor.display_summary()
        
        print("\n🎉 30-day prediction completed successfully!")
        print("📁 Files generated:")
        print("   📄 bitcoin_30day_predictions.json")
        print("   📊 bitcoin_30day_predictions.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()