#!/usr/bin/env python3
"""
IMT Challenge #5 - Bitcoin Price Prediction Demo
Demonstration script for the cryptocurrency investment analysis challenge
"""

import os
import sys
import json
from datetime import datetime

def print_header():
    """Print challenge header"""
    print("=" * 70)
    print("🚀 IMT Challenge #5 - Cryptocurrency Investment Analysis")
    print("   Bitcoin Price Prediction System Demonstration")
    print("=" * 70)
    print(f"📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("👨‍💻 Developer: Amin Haghi")
    print("=" * 70)

def show_challenge_requirements():
    """Show challenge requirements"""
    print("\n📋 IMT Challenge #5 Requirements:")
    print("-" * 50)
    print("✅ 1. تحلیل روندهای رشد و افت قیمت ارزها")
    print("✅ 2. ساخت مدل‌های پیش‌بینی قیمت با یادگیری ماشین")
    print("✅ 3. جمع‌آوری اخبار حوزه کریپتو و تحلیل احساسات بازار")
    print("✅ 4. طراحی پورتفولیوی بهینه بر اساس بازده و ریسک")
    print("-" * 50)

def show_project_features():
    """Show project features"""
    print("\n🛠️ Project Features:")
    print("-" * 50)
    print("📊 Data Sources:")
    print("   • Yahoo Finance (Free, no API required)")
    print("   • Binance API (Real-time data)")
    print()
    print("🤖 Machine Learning Models:")
    print("   • Random Forest Regressor")
    print("   • Linear Regression")
    print("   • PyTorch Neural Networks")
    print()
    print("📈 Technical Analysis:")
    print("   • RSI (Relative Strength Index)")
    print("   • MACD (Moving Average Convergence Divergence)")
    print("   • Bollinger Bands")
    print("   • Multiple Moving Averages")
    print("   • Volume Analysis")
    print()
    print("🎯 Prediction Capabilities:")
    print("   • Short-term (1-day) predictions")
    print("   • Long-term (30-day) predictions")
    print("   • Buy/Hold/Sell signals")
    print("   • Confidence levels (50-95%)")
    print("-" * 50)

def show_sample_output():
    """Show sample prediction output"""
    print("\n📈 Sample Prediction Output:")
    print("-" * 50)
    
    sample_output = {
        "timestamp": "2024-01-15 10:30:00",
        "current_price": 42500.50,
        "predicted_price": 43200.75,
        "price_change": 700.25,
        "price_change_percent": 1.65,
        "confidence": 85.5,
        "signal": "BUY",
        "signal_strength": 0.75,
        "technical_indicators": {
            "rsi": 65.2,
            "macd": 120.5,
            "macd_signal": 115.3,
            "bollinger_position": 0.7,
            "ma_5": 42300.0,
            "ma_20": 41800.0
        },
        "risk_assessment": "MODERATE",
        "recommendation": "Consider buying with proper risk management"
    }
    
    print(json.dumps(sample_output, indent=2, ensure_ascii=False))
    print("-" * 50)

def show_usage_instructions():
    """Show usage instructions"""
    print("\n🚀 How to Run the System:")
    print("-" * 50)
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Run the main menu:")
    print("   python main_menu.py")
    print()
    print("3. Choose your option:")
    print("   • Option 2: Yahoo Finance (Recommended)")
    print("   • Option 4: 30-day Prediction")
    print("   • Option 1: Binance (Requires API keys)")
    print()
    print("4. View results:")
    print("   • JSON output files")
    print("   • Console predictions")
    print("   • Technical analysis charts")
    print("-" * 50)

def show_project_files():
    """Show project files"""
    print("\n📁 Project Files:")
    print("-" * 50)
    
    files = [
        ("main_menu.py", "Main application interface"),
        ("bitcoin_prediction_yahoo.py", "Yahoo Finance predictor"),
        ("bitcoin_30day_predictor.py", "Long-term predictor"),
        ("bitcoin_prediction_system.py", "Binance predictor"),
        ("requirements.txt", "Project dependencies"),
        ("README.md", "Quick start guide"),
        ("COMPLETE_GUIDE.md", "Complete documentation"),
        ("IMT_CHALLENGE_SUBMISSION.md", "Challenge submission"),
        ("IMT_README.md", "Challenge-specific README"),
        ("imt_challenge_demo.py", "This demonstration script")
    ]
    
    for filename, description in files:
        status = "✅" if os.path.exists(filename) else "❌"
        print(f"{status} {filename:<30} - {description}")
    
    print("-" * 50)

def show_challenge_links():
    """Show challenge links"""
    print("\n🔗 Challenge Links:")
    print("-" * 50)
    print("📋 IMT Challenge Repository:")
    print("   https://github.com/MrezaMomeni/IMT/tree/main/ChallengeHub/5-%20Cryptocurrency")
    print()
    print("📊 Dataset (Top 100 Cryptocurrencies):")
    print("   https://www.kaggle.com/datasets/imtkaggleteam/top-100-cryptocurrency-2020-2025/data")
    print()
    print("📱 Telegram Instructions:")
    print("   https://t.me/imtcollege/197")
    print("-" * 50)

def main():
    """Main demonstration function"""
    print_header()
    show_challenge_requirements()
    show_project_features()
    show_sample_output()
    show_usage_instructions()
    show_project_files()
    show_challenge_links()
    
    print("\n🎯 Challenge Status: READY FOR SUBMISSION ✅")
    print("\n👨‍💻 Contact Information:")
    print("   Name: Amin Haghi")
    print("   Email: aminhaghi6@gmail.com")
    print("   Phone: +0034602544560")
    print("\n" + "=" * 70)
    print("🚀 Thank you for reviewing the IMT Challenge #5 submission!")
    print("=" * 70)

if __name__ == "__main__":
    main()