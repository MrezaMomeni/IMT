#!/usr/bin/env python3
"""
Bitcoin Short-term Prediction System - Yahoo Finance Data Source
Short-term Bitcoin prediction using Yahoo Finance data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time

warnings.filterwarnings('ignore')

class BitcoinYahooShortTerm:
    def __init__(self):
        """Initialize Yahoo Finance short-term predictor"""
        self.symbol = "BTC-USD"
        self.scaler = StandardScaler()
        self.model = None
        
        print("🚀 Bitcoin Yahoo Finance Short-term Predictor Initialized")
        print(f"📊 Symbol: {self.symbol}")
        
    def get_current_price(self):
        """Get current Bitcoin price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                print(f"💰 Current BTC Price (Yahoo): ${current_price:,.2f}")
                return current_price
            return None
        except Exception as e:
            print(f"❌ Error getting current price: {str(e)}")
            return None
    
    def get_historical_data(self, period="5d", interval="1h"):
        """Get historical data from Yahoo Finance"""
        try:
            print(f"📥 Fetching historical data...")
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
                
            # Rename columns
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            print(f"✅ Historical data fetched: {len(data)} records")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching historical data: {str(e)}")
            return None
    
    def create_features(self, df):
        """Create technical features for short-term prediction"""
        try:
            # Price features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            
            # Moving averages
            for window in [5, 10, 20]:
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=10).std()
            
            # Lag features
            for lag in [1, 2, 3]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
            
            return df.dropna()
            
        except Exception as e:
            print(f"❌ Error creating features: {str(e)}")
            return df
    
    def train_model(self, df):
        """Train short-term prediction model"""
        try:
            print("🤖 Training short-term model...")
            
            # Select features
            feature_columns = [col for col in df.columns if col not in ['close'] and df[col].dtype in ['float64', 'int64']]
            
            X = df[feature_columns].values
            y = df['close'].values
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            # Test accuracy
            test_pred = self.model.predict(X_test_scaled)
            mae = np.mean(np.abs(test_pred - y_test))
            
            print(f"✅ Model trained - MAE: ${mae:,.2f}")
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            return False
    
    def predict_next_price(self, df):
        """Predict next price"""
        try:
            feature_columns = [col for col in df.columns if col not in ['close'] and df[col].dtype in ['float64', 'int64']]
            latest_features = df[feature_columns].iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            predicted_price = self.model.predict(latest_features_scaled)[0]
            
            # Calculate confidence based on recent volatility
            recent_volatility = df['close'].tail(10).std()
            confidence = max(60, min(90, 85 - (recent_volatility / 100)))
            
            return predicted_price, confidence
            
        except Exception as e:
            print(f"❌ Error predicting price: {str(e)}")
            return None, 0
    
    def generate_trading_signal(self, current_price, predicted_price, confidence):
        """Generate trading signal"""
        try:
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            if abs(price_change_pct) < 0.5:
                signal = "HOLD"
                reason = f"Small predicted change ({price_change_pct:.2f}%)"
            elif price_change_pct > 2 and confidence > 75:
                signal = "BUY"
                reason = f"Strong upward prediction ({price_change_pct:.2f}%)"
            elif price_change_pct < -2 and confidence > 75:
                signal = "SELL"
                reason = f"Strong downward prediction ({price_change_pct:.2f}%)"
            else:
                signal = "HOLD"
                reason = f"Moderate prediction or low confidence"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'predicted_change': price_change_pct
            }
            
        except Exception as e:
            print(f"❌ Error generating signal: {str(e)}")
            return {'signal': 'ERROR', 'confidence': 0, 'reason': str(e)}
    
    def run_prediction_cycle(self):
        """Run one complete prediction cycle"""
        try:
            print("\n" + "="*60)
            print(f"🔄 Yahoo Finance Prediction Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            
            # Get current price
            current_price = self.get_current_price()
            if not current_price:
                print("❌ Failed to get current price")
                return None
            
            # Get historical data
            df = self.get_historical_data(period="7d", interval="1h")
            if df is None or len(df) < 50:
                print("❌ Insufficient historical data")
                return None
            
            # Create features
            df = self.create_features(df)
            
            # Train model
            if not self.train_model(df):
                print("❌ Model training failed")
                return None
            
            # Make prediction
            predicted_price, confidence = self.predict_next_price(df)
            if predicted_price is None:
                print("❌ Prediction failed")
                return None
            
            print(f"🔮 Predicted Price: ${predicted_price:,.2f}")
            print(f"📊 Confidence: {confidence:.1f}%")
            
            # Generate trading signal
            signal_data = self.generate_trading_signal(current_price, predicted_price, confidence)
            
            print(f"\n📈 Trading Signal: {signal_data['signal']}")
            print(f"🎯 Confidence: {signal_data['confidence']:.1f}%")
            print(f"💭 Reason: {signal_data['reason']}")
            
            # Save result
            result = {
                'timestamp': datetime.now().isoformat(),
                'data_source': 'Yahoo Finance',
                'current_price': current_price,
                'predicted_price': predicted_price,
                'confidence': confidence,
                'signal': signal_data
            }
            
            with open('prediction_results_yahoo.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
            
            print(f"💾 Results saved to prediction_results_yahoo.json")
            return result
            
        except Exception as e:
            print(f"❌ Error in prediction cycle: {str(e)}")
            return None
    
    def start_live_monitoring(self, interval_minutes=5, duration_hours=1):
        """Start live monitoring"""
        try:
            print(f"🚀 Starting Live Yahoo Finance Monitoring")
            print(f"⏱️ Interval: {interval_minutes} minutes")
            print(f"⏰ Duration: {duration_hours} hours")
            print("="*60)
            
            start_time = time.time()
            end_time = start_time + (duration_hours * 3600)
            cycle_count = 0
            
            while time.time() < end_time:
                try:
                    cycle_count += 1
                    print(f"\n🔄 Cycle {cycle_count}")
                    
                    result = self.run_prediction_cycle()
                    
                    if result:
                        print(f"✅ Cycle {cycle_count} completed successfully")
                    else:
                        print(f"❌ Cycle {cycle_count} failed")
                    
                    if time.time() < end_time:
                        print(f"⏳ Waiting {interval_minutes} minutes for next cycle...")
                        time.sleep(interval_minutes * 60)
                    
                except KeyboardInterrupt:
                    print("\n⏹️ Monitoring stopped by user")
                    break
                except Exception as e:
                    print(f"❌ Error in monitoring cycle: {str(e)}")
                    time.sleep(30)
            
            print(f"\n🏁 Monitoring completed. Total cycles: {cycle_count}")
            
        except Exception as e:
            print(f"❌ Error in live monitoring: {str(e)}")

def main():
    """Main function with user menu"""
    print("🚀 Bitcoin Yahoo Finance Prediction System")
    print("=" * 50)
    
    system = BitcoinYahooShortTerm()
    
    while True:
        print("\n📋 Yahoo Finance Menu:")
        print("1. Single Prediction Cycle")
        print("2. Start Live Monitoring")
        print("0. Exit")
        
        choice = input("\nSelect option (0-2): ").strip()
        
        if choice == "1":
            print("\n🔄 Running single prediction cycle...")
            system.run_prediction_cycle()
            
        elif choice == "2":
            try:
                duration = int(input("Enter monitoring duration (minutes): "))
                interval = int(input("Enter update interval (minutes): "))
                print(f"\n🔄 Starting live monitoring for {duration} minutes...")
                duration_hours = duration / 60
                system.start_live_monitoring(interval_minutes=interval, duration_hours=duration_hours)
            except ValueError:
                print("❌ Please enter valid numbers")
                
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option! Please try again.")
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()