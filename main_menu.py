#!/usr/bin/env python3
"""
Bitcoin Price Prediction System - Main Menu
Main menu interface for running different prediction methods
"""

import os
import sys
import subprocess
from datetime import datetime

def clear_screen():
    """Clear the screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Display program header"""
    print("=" * 60)
    print("🚀 Bitcoin Price Prediction System")
    print("   Advanced ML-based Bitcoin Price Forecasting")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def print_menu():
    """Display main menu"""
    print("\n📋 Main Menu:")
    print("-" * 40)
    print("1️⃣  Short-term Prediction (Binance)")
    print("2️⃣  Short-term Prediction (Yahoo Finance)")
    print("3️⃣  Yahoo Finance Short-term Predictor")
    print("4️⃣  30-day Prediction")
    print("5️⃣  View 30-day Results")
    print("6️⃣  Test API Keys")
    print("7️⃣  System Summary")
    print("8️⃣  Show Project Files")
    print("9️⃣  Usage Guide")
    print("0️⃣  Exit")
    print("-" * 40)

def run_script(script_name, description):
    """Run script with status message"""
    print(f"\n🔄 Running {description}...")
    print(f"   Executing {script_name}...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(f"\n✅ {description} executed successfully!")
        else:
            print(f"\n❌ Error executing {description}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    input("\n⏸️  Press Enter to continue...")

def show_project_files():
    """Display project files"""
    print("\n📁 Project Files:")
    print("-" * 50)
    
    files = [
        ("main_menu.py", "Main menu system"),
        ("bitcoin_prediction_system.py", "Main prediction system (Binance)"),
        ("bitcoin_prediction_yahoo.py", "Short-term prediction (Yahoo Finance)"),
        ("bitcoin_yahoo_shortterm.py", "Yahoo Finance short-term predictor"),
        ("bitcoin_yahoo_predictor.py", "Yahoo Finance complete predictor"),
        ("bitcoin_30day_predictor.py", "30-day predictor"),
        ("view_30day_predictions.py", "30-day results viewer"),
        ("system_30day_summary.py", "System summary"),
        ("test_api_keys.py", "API keys tester"),
        ("requirements.txt", "Project dependencies"),
        (".env", "API configuration"),
        ("README.md", "Quick start guide"),
        ("COMPLETE_GUIDE.md", "Complete documentation"),
        ("run_menu.bat", "Quick launcher"),
        ("bitcoin_30day_predictions.json", "30-day prediction results"),
        ("prediction_results.json", "Short-term prediction results (Binance)"),
        ("prediction_results_yahoo.json", "Short-term prediction results (Yahoo)"),
        ("bitcoin_30day_detailed_analysis.png", "Analysis chart")
    ]
    
    for i, (filename, description) in enumerate(files, 1):
        status = "✅" if os.path.exists(filename) else "❌"
        print(f"{status} {i:2d}. {filename:<35} - {description}")
    
    input("\n⏸️  Press Enter to continue...")

def show_usage_guide():
    """Display usage guide"""
    print("\n📖 Usage Guide:")
    print("-" * 50)
    print("🔧 Install Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("🔑 Setup API Keys (Optional for Yahoo Finance):")
    print("   Edit .env file and enter your Binance API keys")
    print("   Yahoo Finance options don't need API keys")
    print()
    print("🚀 Run System:")
    print("   1. Start with: python main_menu.py")
    print("   2. For quick start: Option 2 (Yahoo Finance)")
    print("   3. For Binance data: Option 1 (needs API keys)")
    print("   4. For 30-day prediction: Option 4")
    print("   5. Test API keys first: Option 6")
    print()
    print("📊 Data Sources:")
    print("   • Binance API: Real-time data (needs API keys)")
    print("   • Yahoo Finance: Free data (no API needed)")
    print()
    print("📄 Output Files:")
    print("   • prediction_results.json - Binance results")
    print("   • prediction_results_yahoo.json - Yahoo results")
    print("   • bitcoin_30day_predictions.json - 30-day results")
    print()
    print("📚 Documentation:")
    print("   • README.md - Quick start guide")
    print("   • COMPLETE_GUIDE.md - Complete documentation")
    print()
    print("⚠️  Important Notes:")
    print("   • Stable internet connection required")
    print("   • Yahoo Finance is recommended for beginners")
    print("   • Results are for analysis and study only")
    print("   • Never invest based solely on predictions")
    
    input("\n⏸️  Press Enter to continue...")

def main():
    """Main function"""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        try:
            choice = input("\n🔢 Select an option (0-9): ").strip()
            
            if choice == '1':
                run_script("bitcoin_prediction_system.py", "Short-term Prediction (Binance)")
            elif choice == '2':
                run_script("bitcoin_prediction_yahoo.py", "Short-term Prediction (Yahoo Finance)")
            elif choice == '3':
                 run_script("bitcoin_yahoo_shortterm.py", "Yahoo Finance Short-term Predictor")
            elif choice == '4':
                run_script("bitcoin_30day_predictor.py", "30-day Prediction")
            elif choice == '5':
                run_script("view_30day_predictions.py", "30-day Results Viewer")
            elif choice == '6':
                run_script("test_api_keys.py", "API Keys Test")
            elif choice == '7':
                run_script("system_30day_summary.py", "System Summary")
            elif choice == '8':
                show_project_files()
            elif choice == '9':
                show_usage_guide()
            elif choice == '0':
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid option! Please select 0-9")
                input("⏸️  Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()