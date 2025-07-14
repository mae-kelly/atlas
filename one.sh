#!/bin/bash

# AI Trading Empire - Configuration Setup Script
# Creates necessary configuration files and directories

set -e

echo "⚙️  AI Trading Empire - Configuration Setup"
echo "==========================================="

# Create config directory if it doesn't exist
mkdir -p config

echo "📝 Creating API credentials template..."

# Create the API credentials template
cat > config/api_credentials.env << 'EOF'
# AI Trading Empire - API Configuration
# Copy this file to .env and fill in your API keys

# ===========================================
# CRYPTOCURRENCY EXCHANGES
# ===========================================

# Binance (Free - for price data)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_here

# Coinbase Pro (Optional)
COINBASE_API_KEY=your_coinbase_key_here
COINBASE_SECRET=your_coinbase_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# ===========================================
# MARKET DATA PROVIDERS
# ===========================================

# CoinGecko (Free tier available)
COINGECKO_API_KEY=your_coingecko_key_here

# Alpha Vantage (Free tier: 5 calls/min)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Twelve Data (Free tier available)
TWELVE_DATA_API_KEY=your_twelve_data_key_here

# ===========================================
# NEWS & SENTIMENT SOURCES
# ===========================================

# News API (Free tier: 1000 requests/month)
NEWS_API_KEY=your_news_api_key_here

# Twitter API v2 (Essential tier available)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
TWITTER_CONSUMER_KEY=your_twitter_consumer_key_here
TWITTER_CONSUMER_SECRET=your_twitter_consumer_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret_here

# Reddit API (Free)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=AITradingEmpire/1.0

# ===========================================
# OPTIONAL ADVANCED APIS
# ===========================================

# Polygon.io (Free tier available)
POLYGON_API_KEY=your_polygon_key_here

# Finnhub (Free tier available)
FINNHUB_API_KEY=your_finnhub_key_here

# IEX Cloud (Free tier available)
IEX_CLOUD_API_KEY=your_iex_key_here

# FRED Economic Data (Free)
FRED_API_KEY=your_fred_key_here

# ===========================================
# DATABASE & STORAGE (Optional)
# ===========================================

# Database URL (SQLite by default)
DATABASE_URL=sqlite:///data/trading.db

# Redis (Optional - for caching)
REDIS_URL=redis://localhost:6379/0

# ===========================================
# APPLICATION SETTINGS
# ===========================================

# Environment (development/production)
ENVIRONMENT=development

# Debug mode
DEBUG=true

# Log level (DEBUG/INFO/WARNING/ERROR)
LOG_LEVEL=INFO

# ===========================================
# NOTES:
# ===========================================
# 
# REQUIRED FOR BASIC FUNCTIONALITY:
# - None! The system works with free APIs and demo data
#
# RECOMMENDED FOR FULL FEATURES:
# - BINANCE_API_KEY (free account for real price data)
# - NEWS_API_KEY (free tier for sentiment analysis)
#
# OPTIONAL ENHANCEMENTS:
# - TWITTER_BEARER_TOKEN (for social sentiment)
# - ALPHA_VANTAGE_API_KEY (for additional market data)
#
# GET FREE API KEYS:
# - Binance: https://www.binance.com/en/my/settings/api-management
# - News API: https://newsapi.org/register
# - Twitter: https://developer.twitter.com/
# - Alpha Vantage: https://www.alphavantage.co/support/#api-key
# - CoinGecko: https://www.coingecko.com/en/api/pricing
#
EOF

echo "✅ API credentials template created at: config/api_credentials.env"

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists"
    echo "   Backing up existing .env to .env.backup"
    cp .env .env.backup
fi

# Copy template to .env
echo "📋 Copying template to .env..."
cp config/api_credentials.env .env

echo "✅ Configuration file created: .env"

# Create additional config files
echo "📄 Creating additional configuration files..."

# Create trading config
cat > config/trading_config.yaml << 'EOF'
# AI Trading Empire - Trading Configuration

# Portfolio Settings
portfolio:
  initial_capital: 100000.0  # Starting capital in USD
  max_position_size: 0.10    # Maximum 10% per position
  max_portfolio_risk: 0.02   # Maximum 2% daily VaR
  target_sharpe_ratio: 1.5   # Target Sharpe ratio

# Risk Management
risk_management:
  use_kelly_criterion: true
  kelly_fraction: 0.25       # Use 25% of optimal Kelly
  max_drawdown: 0.15         # Stop at 15% drawdown
  confidence_threshold: 0.3  # Minimum prediction confidence

# Data Sources
data_sources:
  price_update_interval: 3   # Seconds between price updates
  sentiment_update_interval: 30  # Seconds between sentiment updates
  correlation_window: 120    # Seconds for correlation analysis

# Machine Learning
ml:
  retrain_interval: 3600     # Retrain models every hour
  prediction_horizon: 15     # Predict 15 minutes ahead
  feature_window: 60         # Use 60 minutes of features
  confidence_threshold: 0.5  # Minimum prediction confidence

# Exchanges
exchanges:
  primary: "binance"
  backup: ["coinbase", "kraken"]
  
# Symbols to Trade
symbols:
  crypto:
    - "BTCUSDT"
    - "ETHUSDT"
    - "ADAUSDT"
    - "SOLUSDT"
  
# Demo Settings
demo:
  duration_minutes: 5        # How long to run demos
  update_frequency: 3        # Seconds between updates
  enable_live_data: false    # Use live data or simulation
EOF

echo "✅ Trading configuration created: config/trading_config.yaml"

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{models,historical,performance,logs}

# Create a simple .env checker script
cat > check_config.py << 'EOF'
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
EOF

chmod +x check_config.py

echo ""
echo "✨ Configuration Setup Complete!"
echo ""
echo "📁 Created files:"
echo "   ✅ config/api_credentials.env (template)"
echo "   ✅ .env (your configuration file)"
echo "   ✅ config/trading_config.yaml"
echo "   ✅ check_config.py (configuration checker)"
echo "   ✅ data/ directories"
echo ""
echo "🚀 Next Steps:"
echo "   1. Edit .env file with your API keys (optional)"
echo "   2. Run: python3 check_config.py"
echo "   3. Run: ./run_simple_live_demo.sh"
echo ""
echo "💡 Pro Tips:"
echo "   - System works WITHOUT API keys (uses demo data)"
echo "   - Add Binance API key for real price data"
echo "   - Add News API key for sentiment analysis"
echo "   - Get free API keys from the URLs in .env comments"
echo ""
echo "✅ You can now run the trading system!"