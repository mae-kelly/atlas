#!/usr/bin/env python3
"""
Quick configuration checker for AI Trading Empire
"""

import os
from dotenv import load_dotenv

def check_config():
    print("🔍 Checking AI Trading Empire Configuration")
    print("=" * 50)
    
    # Load .env file
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("   Run: cp config/api_credentials.env .env")
        return False
    
    load_dotenv()
    
    # Check essential configurations
    essential_vars = {
        'ENVIRONMENT': 'Application environment',
        'DEBUG': 'Debug mode',
        'LOG_LEVEL': 'Logging level'
    }
    
    optional_vars = {
        'BINANCE_API_KEY': 'Binance API (recommended)',
        'NEWS_API_KEY': 'News API (recommended)', 
        'TWITTER_BEARER_TOKEN': 'Twitter API (optional)',
        'ALPHA_VANTAGE_API_KEY': 'Alpha Vantage (optional)',
        'COINGECKO_API_KEY': 'CoinGecko (optional)'
    }
    
    print("📋 Essential Configuration:")
    all_essential_ok = True
    for var, desc in essential_vars.items():
        value = os.getenv(var)
        if value and value != f'your_{var.lower()}_here':
            print(f"   ✅ {desc}: Configured")
        else:
            print(f"   ⚠️  {desc}: Using default")
    
    print("\n🔌 API Keys:")
    configured_apis = 0
    total_apis = len(optional_vars)
    
    for var, desc in optional_vars.items():
        value = os.getenv(var)
        if value and value != f'your_{var.lower()}_here' and 'your_' not in value:
            print(f"   ✅ {desc}: Configured")
            configured_apis += 1
        else:
            print(f"   ⚠️  {desc}: Not configured")
    
    print(f"\n📊 API Configuration: {configured_apis}/{total_apis} configured")
    
    if configured_apis == 0:
        print("\n💡 System will work with demo data!")
        print("   - No API keys required for basic functionality")
        print("   - Add API keys for live data and enhanced features")
    elif configured_apis < total_apis // 2:
        print("\n🚀 Basic functionality available!")
        print("   - Add more API keys for full features")
    else:
        print("\n🏆 Full functionality available!")
    
    print(f"\n📁 Data Directories:")
    data_dirs = ['data/models', 'data/historical', 'data/performance', 'data/logs']
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✅ {dir_path} (created)")
    
    print("\n✨ Configuration check complete!")
    print("\n🚀 Ready to run:")
    print("   ./run_simple_live_demo.sh")
    
    return True

if __name__ == "__main__":
    check_config()
