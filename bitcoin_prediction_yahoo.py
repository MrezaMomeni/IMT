#!/usr/bin/env python3
"""
Bitcoin Price Prediction System - Yahoo Finance Data Source
Advanced Bitcoin price prediction using Yahoo Finance data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BitcoinPredictionYahoo:
    def __init__(self):
        """Initialize Bitcoin Prediction System with Yahoo Finance"""
        self.symbol = "BTC-USD"
        self.sequence_length = 60
        print("🚀 Bitcoin Prediction System Initialized (Yahoo Finance)")
        print("📊 Data Source: Yahoo Finance")
        
    def get_current_price(self):
        """Get current Bitcoin price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                print(f"✅ Current BTC Price: ${current_price:,.2f}")
                return float(current_price)
            return None
        except Exception as e:
            print(f"❌ Error getting current price: {str(e)}")
            return None
    
    def get_historical_data(self, period="30d", interval="1h"):
        """Get historical Bitcoin data from Yahoo Finance"""
        try:
            print(f"📥 Fetching historical data (Period: {period}, Interval: {interval})...")
            
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print("❌ No data received from Yahoo Finance")
                return None
            
            # Convert to our format
            df = data.copy()
            df.reset_index(inplace=True)
            
            # Rename columns to lowercase
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Datetime': 'timestamp'
            })
            
            # If Datetime column doesn't exist, use the index
            if 'timestamp' not in df.columns and 'Datetime' in df.columns:
                df['timestamp'] = df['Datetime']
            elif 'timestamp' not in df.columns:
                df['timestamp'] = df.index
            
            print(f"✅ Retrieved {len(df)} data points")
            print(f"📊 Columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            return None
    
    def create_features(self, df):
        """Create technical features for prediction"""
        try:
            print("🔧 Creating technical features...")
            
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'price_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
            
            # RSI
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            df['rsi'] = calculate_rsi(df['close'])
            
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
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std()
            
            # Time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            print(f"✅ Created {len(df.columns)} features")
            return df
            
        except Exception as e:
            print(f"❌ Error creating features: {str(e)}")
            return df
    
    def simple_prediction(self, df, method='trend_analysis'):
        """Simple prediction using trend analysis"""
        try:
            if len(df) < 20:
                return None, 0
            
            recent_data = df.tail(20)
            current_price = recent_data['close'].iloc[-1]
            
            if method == 'trend_analysis':
                # Analyze recent trend
                short_ma = recent_data['close'].tail(5).mean()
                long_ma = recent_data['close'].tail(10).mean()
                
                # Price momentum
                price_change_5 = (current_price - recent_data['close'].iloc[-6]) / recent_data['close'].iloc[-6]
                price_change_10 = (current_price - recent_data['close'].iloc[-11]) / recent_data['close'].iloc[-11]
                
                # RSI analysis
                current_rsi = recent_data['rsi'].iloc[-1] if 'rsi' in recent_data.columns else 50
                
                # MACD analysis
                macd_signal = 0
                if 'macd' in recent_data.columns and 'macd_signal' in recent_data.columns:
                    macd_diff = recent_data['macd'].iloc[-1] - recent_data['macd_signal'].iloc[-1]
                    macd_signal = 1 if macd_diff > 0 else -1
                
                # Combine signals
                trend_signal = 1 if short_ma > long_ma else -1
                momentum_signal = 1 if price_change_5 > 0 else -1
                rsi_signal = 1 if current_rsi < 70 else -1  # Not overbought
                
                # Calculate prediction
                signals = [trend_signal, momentum_signal, rsi_signal, macd_signal]
                signal_strength = sum(signals) / len(signals)
                
                # Predict price change
                base_change = np.mean([price_change_5, price_change_10])
                predicted_change = base_change * (1 + signal_strength * 0.1)
                
                predicted_price = current_price * (1 + predicted_change)
                confidence = max(50, min(95, abs(signal_strength) * 80 + 50))
                
                return predicted_price, confidence
            
            elif method == 'moving_average':
                # Simple moving average prediction
                ma_5 = recent_data['close'].tail(5).mean()
                ma_10 = recent_data['close'].tail(10).mean()
                predicted_price = (ma_5 + ma_10) / 2
                
                # Calculate confidence based on price stability
                volatility = recent_data['close'].std() / recent_data['close'].mean()
                confidence = max(50, 90 - volatility * 1000)
                
                return predicted_price, confidence
            
            return current_price, 50
            
        except Exception as e:
            print(f"❌ Error in prediction: {str(e)}")
            return None, 0
    
    def generate_trading_signal(self, current_price, predicted_price, confidence):
        """Generate trading signal based on prediction"""
        try:
            if not current_price or not predicted_price:
                return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
            
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Signal thresholds
            strong_buy_threshold = 2.0
            buy_threshold = 0.5
            sell_threshold = -0.5
            strong_sell_threshold = -2.0
            
            # Determine signal
            if price_change_pct >= strong_buy_threshold and confidence >= 70:
                signal = 'STRONG_BUY'
                reason = f'Strong upward trend predicted (+{price_change_pct:.2f}%)'
            elif price_change_pct >= buy_threshold and confidence >= 60:
                signal = 'BUY'
                reason = f'Upward trend predicted (+{price_change_pct:.2f}%)'
            elif price_change_pct <= strong_sell_threshold and confidence >= 70:
                signal = 'STRONG_SELL'
                reason = f'Strong downward trend predicted ({price_change_pct:.2f}%)'
            elif price_change_pct <= sell_threshold and confidence >= 60:
                signal = 'SELL'
                reason = f'Downward trend predicted ({price_change_pct:.2f}%)'
            else:
                signal = 'HOLD'
                reason = f'Sideways movement predicted ({price_change_pct:.2f}%)'
            
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            if signal in ['BUY', 'STRONG_BUY']:
                stop_loss = current_price * 0.98  # 2% stop loss
                take_profit = current_price * 1.04  # 4% take profit
            elif signal in ['SELL', 'STRONG_SELL']:
                stop_loss = current_price * 1.02  # 2% stop loss for short
                take_profit = current_price * 0.96  # 4% take profit for short
            else:
                stop_loss = None
                take_profit = None
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'predicted_change': price_change_pct
            }
            
        except Exception as e:
            print(f"❌ Error generating signal: {str(e)}")
            return {'signal': 'ERROR', 'confidence': 0, 'reason': str(e)}
    
    def run_prediction_cycle(self):
        """Run one complete prediction cycle"""
        try:
            print("\n" + "="*60)
            print(f"🔄 Running Prediction Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("📊 Data Source: Yahoo Finance")
            print("="*60)
            
            current_price = self.get_current_price()
            if not current_price:
                print("❌ Failed to get current price")
                return None
            
            df = self.get_historical_data(period="30d", interval="1h")
            if df is None or len(df) < self.sequence_length:
                print("❌ Insufficient historical data")
                return None
            
            df = self.create_features(df)
            
            predicted_price, confidence = self.simple_prediction(df, method='trend_analysis')
            if predicted_price is None:
                print("❌ Prediction failed")
                return None
            
            print(f"🔮 Predicted Price: ${predicted_price:,.2f}")
            print(f"📊 Confidence: {confidence:.1f}%")
            
            signal_data = self.generate_trading_signal(current_price, predicted_price, confidence)
            
            print(f"\n📈 Trading Signal: {signal_data['signal']}")
            print(f"🎯 Confidence: {signal_data['confidence']:.1f}%")
            print(f"💭 Reason: {signal_data['reason']}")
            
            if signal_data['entry_price']:
                print(f"💵 Entry Price: ${signal_data['entry_price']:,.2f}")
                if signal_data['stop_loss']:
                    print(f"🛑 Stop Loss: ${signal_data['stop_loss']:,.2f}")
                if signal_data['take_profit']:
                    print(f"🎯 Take Profit: ${signal_data['take_profit']:,.2f}")
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'data_source': 'Yahoo Finance',
                'current_price': current_price,
                'predicted_price': predicted_price,
                'confidence': confidence,
                'signal': signal_data
            }
            
            # Save results
            filename = 'prediction_results_yahoo.json'
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
            
            print(f"💾 Results saved to {filename}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in prediction cycle: {str(e)}")
            return None
    
    def start_live_monitoring(self, interval_minutes=5, duration_hours=1):
        """Start live monitoring and prediction"""
        try:
            print(f"🚀 Starting Live Bitcoin Prediction System (Yahoo Finance)")
            print(f"⏱️ Interval: {interval_minutes} minutes")
            print(f"⏰ Duration: {duration_hours} hours")
            print(f"🛑 Press Ctrl+C to stop")
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
                    print("⏳ Waiting 30 seconds before retry...")
                    time.sleep(30)
            
            print(f"\n🏁 Monitoring completed. Total cycles: {cycle_count}")
            
        except Exception as e:
            print(f"❌ Error in live monitoring: {str(e)}")

def main():
    """Main function with user menu"""
    print("🚀 Bitcoin Price Prediction System (Yahoo Finance)")
    print("=" * 50)
    
    system = BitcoinPredictionYahoo()
    
    while True:
        print("\n📋 Main Menu:")
        print("1. Single Prediction Cycle")
        print("2. Start Live Monitoring")
        print("3. Test Data Connection")
        print("0. Exit")
        
        choice = input("\nSelect option (0-3): ").strip()
        
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
                
        elif choice == "3":
            print("\n🔍 Testing data connection...")
            price = system.get_current_price()
            if price:
                print("✅ Yahoo Finance connection successful!")
            else:
                print("❌ Yahoo Finance connection failed!")
                
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option! Please try again.")
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()