#!/bin/bash

# AI Trading Empire - US Live Trading Setup (Part 2)
# Exchange API Setup, Testing & Live Trading Activation
# Configures live trading with US-compliant exchanges

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

header() {
    echo -e "${PURPLE}"
    echo "======================================================="
    echo "$1"
    echo "======================================================="
    echo -e "${NC}"
}

success() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    header "CHECKING PREREQUISITES"
    
    # Check if Part 1 was completed
    if [[ ! -d "trading_env" ]]; then
        error "Virtual environment not found. Please run Part 1 setup first."
        exit 1
    fi
    
    # Activate virtual environment
    source trading_env/bin/activate
    log "Virtual environment activated ✅"
    
    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.us_template" ]]; then
            warn ".env file not found. Copying from template..."
            cp .env.us_template .env
            warn "⚠️  IMPORTANT: Edit .env file with your actual API keys before proceeding!"
            echo
            read -p "Have you configured your API keys in .env? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                warn "Please configure your API keys in .env file first:"
                echo "1. Edit .env file with your exchange API keys"
                echo "2. Add your data provider API keys"
                echo "3. Re-run this script"
                exit 1
            fi
        else
            error ".env template not found. Please run Part 1 setup first."
            exit 1
        fi
    fi
    
    log "Prerequisites check completed ✅"
}

# Test critical Python imports
test_imports() {
    header "TESTING PYTHON IMPORTS"
    
    log "Testing core trading imports..."
    
    python -c "
import sys
import traceback

def test_import(module_name, description):
    try:
        __import__(module_name)
        print(f'✅ {description}')
        return True
    except ImportError as e:
        print(f'❌ {description}: {e}')
        return False

# Test critical imports
critical_imports = [
    ('ccxt', 'CCXT exchange library'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('aiohttp', 'Async HTTP client'),
    ('python_dotenv', 'Environment variables'),
    ('loguru', 'Logging'),
]

optional_imports = [
    ('talib', 'TA-Lib technical analysis'),
    ('ta', 'Alternative TA library'),
    ('pandas_ta', 'Pandas TA'),
    ('xgboost', 'XGBoost'),
    ('lightgbm', 'LightGBM'),
    ('catboost', 'CatBoost'),
    ('sklearn', 'Scikit-learn'),
    ('torch', 'PyTorch'),
]

print('Critical imports:')
critical_success = all(test_import(module, desc) for module, desc in critical_imports)

print('\nOptional imports:')
for module, desc in optional_imports:
    test_import(module, desc)

if not critical_success:
    print('\n❌ Critical imports failed. Please check Part 1 installation.')
    sys.exit(1)
else:
    print('\n✅ All critical imports successful!')
"
    
    log "Python imports test completed ✅"
}

# Create exchange connector test script
create_exchange_test_script() {
    header "CREATING EXCHANGE TEST SCRIPT"
    
    cat > test_exchanges.py << 'EOF'
#!/usr/bin/env python3
"""
US Exchange API Connection Test Script
Tests connections to US-compliant cryptocurrency exchanges
"""

import asyncio
import ccxt
import os
import sys
from dotenv import load_dotenv
import json
from datetime import datetime
import aiohttp

# Load environment variables
load_dotenv()

class ExchangeTester:
    def __init__(self):
        self.results = {}
        self.exchanges = {}
        
    async def test_coinbase_pro(self):
        """Test Coinbase Pro/Advanced API connection"""
        print("🧪 Testing Coinbase Pro API...")
        
        api_key = os.getenv('COINBASE_API_KEY')
        secret = os.getenv('COINBASE_SECRET')
        passphrase = os.getenv('COINBASE_PASSPHRASE')
        
        if not all([api_key, secret, passphrase]):
            self.results['coinbase_pro'] = {
                'status': 'skipped',
                'message': 'API credentials not configured'
            }
            print("⚠️  Coinbase Pro: API credentials not configured")
            return
            
        try:
            exchange = ccxt.coinbasepro({
                'apiKey': api_key,
                'secret': secret,
                'password': passphrase,
                'sandbox': False,  # Use real API for testing
                'enableRateLimit': True,
            })
            
            # Test public API first
            ticker = await exchange.fetch_ticker('BTC/USD')
            print(f"✅ Coinbase Pro: Public API working - BTC price: ${ticker['last']:,.2f}")
            
            # Test private API (account info)
            try:
                balance = await exchange.fetch_balance()
                print(f"✅ Coinbase Pro: Private API working - Account connected")
                
                self.results['coinbase_pro'] = {
                    'status': 'success',
                    'message': 'Full API access confirmed',
                    'btc_price': ticker['last'],
                    'account_currencies': len([k for k, v in balance.items() if isinstance(v, dict) and v.get('total', 0) > 0])
                }
                self.exchanges['coinbase_pro'] = exchange
                
            except Exception as e:
                print(f"⚠️  Coinbase Pro: Private API failed - {e}")
                self.results['coinbase_pro'] = {
                    'status': 'partial',
                    'message': f'Public API works, Private API failed: {e}',
                    'btc_price': ticker['last']
                }
                
        except Exception as e:
            print(f"❌ Coinbase Pro: Connection failed - {e}")
            self.results['coinbase_pro'] = {
                'status': 'failed',
                'message': str(e)
            }
    
    async def test_kraken(self):
        """Test Kraken API connection"""
        print("🧪 Testing Kraken API...")
        
        api_key = os.getenv('KRAKEN_API_KEY')
        secret = os.getenv('KRAKEN_SECRET')
        
        if not all([api_key, secret]):
            self.results['kraken'] = {
                'status': 'skipped',
                'message': 'API credentials not configured'
            }
            print("⚠️  Kraken: API credentials not configured")
            return
            
        try:
            exchange = ccxt.kraken({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
            })
            
            # Test public API
            ticker = await exchange.fetch_ticker('BTC/USD')
            print(f"✅ Kraken: Public API working - BTC price: ${ticker['last']:,.2f}")
            
            # Test private API
            try:
                balance = await exchange.fetch_balance()
                print(f"✅ Kraken: Private API working - Account connected")
                
                self.results['kraken'] = {
                    'status': 'success',
                    'message': 'Full API access confirmed',
                    'btc_price': ticker['last'],
                    'account_currencies': len([k for k, v in balance.items() if isinstance(v, dict) and v.get('total', 0) > 0])
                }
                self.exchanges['kraken'] = exchange
                
            except Exception as e:
                print(f"⚠️  Kraken: Private API failed - {e}")
                self.results['kraken'] = {
                    'status': 'partial',
                    'message': f'Public API works, Private API failed: {e}',
                    'btc_price': ticker['last']
                }
                
        except Exception as e:
            print(f"❌ Kraken: Connection failed - {e}")
            self.results['kraken'] = {
                'status': 'failed',
                'message': str(e)
            }
    
    async def test_gemini(self):
        """Test Gemini API connection"""
        print("🧪 Testing Gemini API...")
        
        api_key = os.getenv('GEMINI_API_KEY')
        secret = os.getenv('GEMINI_SECRET')
        
        if not all([api_key, secret]):
            self.results['gemini'] = {
                'status': 'skipped',
                'message': 'API credentials not configured'
            }
            print("⚠️  Gemini: API credentials not configured")
            return
            
        try:
            exchange = ccxt.gemini({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
            })
            
            # Test public API
            ticker = await exchange.fetch_ticker('BTC/USD')
            print(f"✅ Gemini: Public API working - BTC price: ${ticker['last']:,.2f}")
            
            # Test private API
            try:
                balance = await exchange.fetch_balance()
                print(f"✅ Gemini: Private API working - Account connected")
                
                self.results['gemini'] = {
                    'status': 'success',
                    'message': 'Full API access confirmed',
                    'btc_price': ticker['last'],
                    'account_currencies': len([k for k, v in balance.items() if isinstance(v, dict) and v.get('total', 0) > 0])
                }
                self.exchanges['gemini'] = exchange
                
            except Exception as e:
                print(f"⚠️  Gemini: Private API failed - {e}")
                self.results['gemini'] = {
                    'status': 'partial',
                    'message': f'Public API works, Private API failed: {e}',
                    'btc_price': ticker['last']
                }
                
        except Exception as e:
            print(f"❌ Gemini: Connection failed - {e}")
            self.results['gemini'] = {
                'status': 'failed',
                'message': str(e)
            }
    
    async def test_data_providers(self):
        """Test data provider APIs"""
        print("\n🧪 Testing Data Provider APIs...")
        
        # Test Alpha Vantage
        alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_key:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={alpha_key}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'Global Quote' in data:
                                print("✅ Alpha Vantage: API working")
                                self.results['alpha_vantage'] = {'status': 'success'}
                            else:
                                print("⚠️  Alpha Vantage: API key may be invalid or rate limited")
                                self.results['alpha_vantage'] = {'status': 'partial', 'message': 'Rate limited or invalid key'}
                        else:
                            print(f"❌ Alpha Vantage: HTTP {response.status}")
                            self.results['alpha_vantage'] = {'status': 'failed', 'message': f'HTTP {response.status}'}
            except Exception as e:
                print(f"❌ Alpha Vantage: {e}")
                self.results['alpha_vantage'] = {'status': 'failed', 'message': str(e)}
        else:
            print("⚠️  Alpha Vantage: API key not configured")
            self.results['alpha_vantage'] = {'status': 'skipped'}
        
        # Test News API
        news_key = os.getenv('NEWS_API_KEY')
        if news_key:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey={news_key}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('status') == 'ok':
                                print("✅ News API: API working")
                                self.results['news_api'] = {'status': 'success'}
                            else:
                                print("⚠️  News API: API response error")
                                self.results['news_api'] = {'status': 'partial'}
                        else:
                            print(f"❌ News API: HTTP {response.status}")
                            self.results['news_api'] = {'status': 'failed', 'message': f'HTTP {response.status}'}
            except Exception as e:
                print(f"❌ News API: {e}")
                self.results['news_api'] = {'status': 'failed', 'message': str(e)}
        else:
            print("⚠️  News API: API key not configured")
            self.results['news_api'] = {'status': 'skipped'}
    
    async def run_tests(self):
        """Run all API tests"""
        print("🚀 Starting US Exchange API Tests")
        print("=" * 50)
        
        # Test exchanges
        await self.test_coinbase_pro()
        await self.test_kraken()
        await self.test_gemini()
        
        # Test data providers
        await self.test_data_providers()
        
        # Close all exchange connections
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'close'):
                await exchange.close()
        
        return self.results
    
    def generate_report(self):
        """Generate test results report"""
        print("\n" + "=" * 50)
        print("🏆 API CONNECTION TEST RESULTS")
        print("=" * 50)
        
        working_exchanges = []
        for name, result in self.results.items():
            if 'coinbase' in name or 'kraken' in name or 'gemini' in name:
                status_icon = "✅" if result['status'] == 'success' else "⚠️" if result['status'] == 'partial' else "❌"
                print(f"{status_icon} {name.replace('_', ' ').title()}: {result['message']}")
                if result['status'] in ['success', 'partial']:
                    working_exchanges.append(name)
        
        print(f"\n📊 Summary:")
        print(f"   Working Exchanges: {len(working_exchanges)}")
        if working_exchanges:
            print(f"   Available for Trading: {', '.join(working_exchanges)}")
            print(f"   ✅ Ready for live trading setup!")
        else:
            print(f"   ❌ No exchanges configured - please add API keys")
        
        # Save results to file
        with open('data/exchange_test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.results,
                'working_exchanges': working_exchanges
            }, f, indent=2)
        
        return len(working_exchanges) > 0

async def main():
    tester = ExchangeTester()
    await tester.run_tests()
    return tester.generate_report()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
EOF

    chmod +x test_exchanges.py
    log "Exchange test script created ✅"
}

# Test exchange connections
test_exchange_connections() {
    header "TESTING EXCHANGE CONNECTIONS"
    
    log "Running exchange API tests..."
    
    if python test_exchanges.py; then
        success "Exchange connection tests completed successfully! ✅"
        return 0
    else
        warn "Some exchange tests failed. Please check your API configuration."
        return 1
    fi
}

# Create live trading management script
create_trading_manager() {
    header "CREATING LIVE TRADING MANAGER"
    
    cat > trading_manager.py << 'EOF'
#!/usr/bin/env python3
"""
Live Trading Manager
Controls paper trading and live trading modes
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
import ccxt
import pandas as pd
import numpy as np
from loguru import logger

load_dotenv()

class TradingManager:
    def __init__(self):
        self.live_trading_enabled = os.getenv('LIVE_TRADING_ENABLED', 'false').lower() == 'true'
        self.paper_trading_mode = os.getenv('PAPER_TRADING_MODE', 'true').lower() == 'true'
        self.primary_exchange = os.getenv('PRIMARY_EXCHANGE', 'coinbase_pro')
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.05'))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', '500'))
        
        self.exchanges = {}
        self.positions = {}
        self.daily_pnl = 0.0
        self.trade_count = 0
        
        # Setup logging
        logger.add("data/logs/trading_{time}.log", rotation="1 day", retention="30 days")
        
    async def initialize_exchanges(self):
        """Initialize exchange connections"""
        logger.info("Initializing exchange connections...")
        
        # Coinbase Pro
        if all([os.getenv('COINBASE_API_KEY'), os.getenv('COINBASE_SECRET'), os.getenv('COINBASE_PASSPHRASE')]):
            self.exchanges['coinbase_pro'] = ccxt.coinbasepro({
                'apiKey': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET'),
                'password': os.getenv('COINBASE_PASSPHRASE'),
                'sandbox': self.paper_trading_mode,
                'enableRateLimit': True,
            })
            logger.info("✅ Coinbase Pro configured")
        
        # Kraken
        if all([os.getenv('KRAKEN_API_KEY'), os.getenv('KRAKEN_SECRET')]):
            self.exchanges['kraken'] = ccxt.kraken({
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_SECRET'),
                'enableRateLimit': True,
            })
            logger.info("✅ Kraken configured")
        
        # Load markets for all exchanges
        for name, exchange in self.exchanges.items():
            try:
                await exchange.load_markets()
                logger.info(f"✅ {name} markets loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load {name} markets: {e}")
    
    async def get_account_status(self):
        """Get account status from primary exchange"""
        if self.primary_exchange not in self.exchanges:
            logger.error(f"Primary exchange {self.primary_exchange} not configured")
            return None
        
        exchange = self.exchanges[self.primary_exchange]
        try:
            balance = await exchange.fetch_balance()
            
            # Calculate total USD value
            total_usd = 0
            for currency, amount in balance.items():
                if isinstance(amount, dict) and amount.get('total', 0) > 0:
                    if currency == 'USD':
                        total_usd += amount['total']
                    else:
                        try:
                            ticker = await exchange.fetch_ticker(f"{currency}/USD")
                            total_usd += amount['total'] * ticker['last']
                        except:
                            pass  # Skip if can't get price
            
            return {
                'exchange': self.primary_exchange,
                'total_usd_value': total_usd,
                'currencies': {k: v for k, v in balance.items() if isinstance(v, dict) and v.get('total', 0) > 0},
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting account status: {e}")
            return None
    
    async def place_order(self, symbol, side, amount, order_type='market', price=None):
        """Place a trading order"""
        if not self.live_trading_enabled and not self.paper_trading_mode:
            logger.warning("Trading not enabled")
            return None
        
        # Risk checks
        if amount > self.max_position_size:
            logger.warning(f"Order size {amount} exceeds maximum {self.max_position_size}")
            return None
        
        if abs(self.daily_pnl) > self.max_daily_loss:
            logger.warning(f"Daily loss limit {self.max_daily_loss} exceeded")
            return None
        
        exchange = self.exchanges.get(self.primary_exchange)
        if not exchange:
            logger.error("Primary exchange not available")
            return None
        
        try:
            if self.paper_trading_mode:
                # Paper trading simulation
                ticker = await exchange.fetch_ticker(symbol)
                simulated_price = ticker['last']
                
                order = {
                    'id': f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'type': order_type,
                    'price': simulated_price,
                    'status': 'closed',
                    'filled': amount,
                    'timestamp': datetime.now().isoformat(),
                    'paper_trading': True
                }
                
                logger.info(f"📄 PAPER TRADE: {side} {amount} {symbol} @ ${simulated_price:.2f}")
                
            else:
                # Real trading
                if order_type == 'market':
                    order = await exchange.create_market_order(symbol, side, amount)
                else:
                    order = await exchange.create_limit_order(symbol, side, amount, price)
                
                logger.info(f"💰 LIVE TRADE: {side} {amount} {symbol}")
            
            self.trade_count += 1
            
            # Save trade to log
            with open('data/logs/trades.json', 'a') as f:
                f.write(json.dumps(order) + '\n')
            
            return order
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None
    
    async def get_market_data(self, symbol):
        """Get current market data"""
        exchange = self.exchanges.get(self.primary_exchange)
        if not exchange:
            return None
        
        try:
            ticker = await exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'change': ticker['percentage'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def get_status(self):
        """Get trading manager status"""
        return {
            'live_trading_enabled': self.live_trading_enabled,
            'paper_trading_mode': self.paper_trading_mode,
            'primary_exchange': self.primary_exchange,
            'exchanges_configured': list(self.exchanges.keys()),
            'daily_pnl': self.daily_pnl,
            'trade_count': self.trade_count,
            'max_position_size': self.max_position_size,
            'max_daily_loss': self.max_daily_loss
        }
    
    async def close(self):
        """Close all exchange connections"""
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'close'):
                await exchange.close()

async def main():
    """Test the trading manager"""
    manager = TradingManager()
    
    try:
        await manager.initialize_exchanges()
        
        print("🏆 Trading Manager Status:")
        print("=" * 40)
        status = manager.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n💰 Account Status:")
        account = await manager.get_account_status()
        if account:
            print(f"  Exchange: {account['exchange']}")
            print(f"  Total USD Value: ${account['total_usd_value']:,.2f}")
            print(f"  Currencies: {len(account['currencies'])}")
        
        print("\n📊 Market Data Test:")
        btc_data = await manager.get_market_data('BTC/USD')
        if btc_data:
            print(f"  BTC/USD: ${btc_data['price']:,.2f} (24h: {btc_data['change']:.2f}%)")
        
        # Test paper trade
        if manager.paper_trading_mode:
            print("\n📄 Testing Paper Trade:")
            order = await manager.place_order('BTC/USD', 'buy', 0.001)
            if order:
                print(f"  ✅ Paper trade successful: {order['id']}")
        
    except Exception as e:
        logger.error(f"Trading manager test failed: {e}")
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(main())
EOF

    chmod +x trading_manager.py
    log "Trading manager script created ✅"
}

# Create paper trading test
create_paper_trading_test() {
    header "CREATING PAPER TRADING TEST"
    
    cat > paper_trading_test.py << 'EOF'
#!/usr/bin/env python3
"""
Paper Trading Test Script
Tests the complete trading pipeline in paper mode
"""

import asyncio
import sys
from trading_manager import TradingManager
from loguru import logger

async def run_paper_trading_test():
    """Run comprehensive paper trading test"""
    print("🧪 Starting Paper Trading Test")
    print("=" * 50)
    
    manager = TradingManager()
    
    try:
        # Initialize
        await manager.initialize_exchanges()
        
        if not manager.exchanges:
            print("❌ No exchanges configured. Please set up API keys.")
            return False
        
        print("✅ Exchanges initialized")
        
        # Test account access
        account = await manager.get_account_status()
        if account:
            print(f"✅ Account connected: ${account['total_usd_value']:,.2f} total value")
        else:
            print("⚠️  Account status unavailable")
        
        # Test market data
        symbols = ['BTC/USD', 'ETH/USD']
        market_data = {}
        
        for symbol in symbols:
            data = await manager.get_market_data(symbol)
            if data:
                market_data[symbol] = data
                print(f"✅ Market data {symbol}: ${data['price']:,.2f}")
            else:
                print(f"❌ Market data failed for {symbol}")
        
        if not market_data:
            print("❌ No market data available")
            return False
        
        # Test paper trades
        print("\n📄 Testing Paper Trades:")
        
        test_trades = [
            ('BTC/USD', 'buy', 0.001),
            ('ETH/USD', 'buy', 0.01),
            ('BTC/USD', 'sell', 0.0005),
        ]
        
        successful_trades = 0
        for symbol, side, amount in test_trades:
            order = await manager.place_order(symbol, side, amount)
            if order:
                print(f"✅ {side.upper()} {amount} {symbol} @ ${order['price']:.2f}")
                successful_trades += 1
            else:
                print(f"❌ Failed to place {side} order for {symbol}")
            
            # Small delay between trades
            await asyncio.sleep(1)
        
        print(f"\n🏆 Paper Trading Test Results:")
        print(f"   Total test trades: {len(test_trades)}")
        print(f"   Successful trades: {successful_trades}")
        print(f"   Success rate: {successful_trades/len(test_trades)*100:.1f}%")
        
        if successful_trades == len(test_trades):
            print("✅ All paper trades successful - System ready for live trading!")
            return True
        else:
            print("⚠️  Some paper trades failed - Check configuration")
            return False
            
    except Exception as e:
        logger.error(f"Paper trading test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        await manager.close()

if __name__ == "__main__":
    success = asyncio.run(run_paper_trading_test())
    print("\n" + "=" * 50)
    if success:
        print("🎉 PAPER TRADING TEST PASSED!")
        print("🚀 Ready to enable live trading when desired")
    else:
        print("❌ PAPER TRADING TEST FAILED")
        print("🔧 Please check your configuration and try again")
    
    sys.exit(0 if success else 1)
EOF

    chmod +x paper_trading_test.py
    log "Paper trading test script created ✅"
}

# Create live trading activation script
create_live_trading_activation() {
    header "CREATING LIVE TRADING ACTIVATION SCRIPT"
    
    cat > activate_live_trading.py << 'EOF'
#!/usr/bin/env python3
"""
Live Trading Activation Script
Safely enables live trading with confirmation and safety checks
"""

import os
import sys
from dotenv import load_dotenv, set_key
import asyncio
from trading_manager import TradingManager

def confirm_live_trading():
    """Get user confirmation for live trading"""
    print("⚠️  " + "=" * 60)
    print("⚠️  LIVE TRADING ACTIVATION")
    print("⚠️  " + "=" * 60)
    print("⚠️  ")
    print("⚠️  YOU ARE ABOUT TO ENABLE LIVE TRADING WITH REAL MONEY")
    print("⚠️  ")
    print("⚠️  This will:")
    print("⚠️  - Execute real trades on exchanges")
    print("⚠️  - Use your actual account funds")
    print("⚠️  - Incur trading fees")
    print("⚠️  - Risk capital loss")
    print("⚠️  ")
    print("⚠️  Make sure you have:")
    print("⚠️  - Tested thoroughly with paper trading")
    print("⚠️  - Set appropriate position limits")
    print("⚠️  - Configured stop-loss settings")
    print("⚠️  - Only invested money you can afford to lose")
    print("⚠️  ")
    print("⚠️  " + "=" * 60)
    
    while True:
        response = input("\nDo you want to enable live trading? Type 'ENABLE LIVE TRADING' to confirm: ")
        if response == "ENABLE LIVE TRADING":
            return True
        elif response.lower() in ['no', 'n', 'exit', 'quit']:
            return False
        else:
            print("Please type exactly 'ENABLE LIVE TRADING' to confirm, or 'no' to cancel.")

async def run_safety_checks():
    """Run safety checks before enabling live trading"""
    print("\n🔒 Running Pre-Activation Safety Checks...")
    
    # Check environment variables
    load_dotenv()
    
    required_vars = [
        'COINBASE_API_KEY', 'COINBASE_SECRET', 'COINBASE_PASSPHRASE',
        'MAX_POSITION_SIZE', 'MAX_DAILY_LOSS'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    # Check position limits
    max_position = float(os.getenv('MAX_POSITION_SIZE', '0.0'))
    if max_position > 0.1:  # 10%
        print(f"⚠️  WARNING: Position size limit is high ({max_position*100:.1f}%)")
        confirm = input("Continue with high position limit? (y/N): ")
        if confirm.lower() != 'y':
            return False
    
    # Test exchange connection
    try:
        manager = TradingManager()
        await manager.initialize_exchanges()
        
        if not manager.exchanges:
            print("❌ No exchange connections available")
            return False
        
        # Test account access
        account = await manager.get_account_status()
        if not account:
            print("❌ Cannot access account information")
            return False
        
        print(f"✅ Account connected: ${account['total_usd_value']:,.2f} available")
        
        await manager.close()
        
    except Exception as e:
        print(f"❌ Exchange connection test failed: {e}")
        return False
    
    print("✅ All safety checks passed")
    return True

def enable_live_trading():
    """Enable live trading in environment"""
    env_file = '.env'
    
    # Update environment variables
    set_key(env_file, 'LIVE_TRADING_ENABLED', 'true')
    set_key(env_file, 'PAPER_TRADING_MODE', 'false')
    
    print("✅ Live trading enabled in .env file")
    print("✅ Paper trading mode disabled")

def create_backup():
    """Create backup of current configuration"""
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f".env.backup_{timestamp}"
    
    shutil.copy('.env', backup_file)
    print(f"✅ Configuration backed up to {backup_file}")

async def main():
    """Main activation process"""
    print("🚀 Live Trading Activation Process")
    print("=" * 50)
    
    # Step 1: User confirmation
    if not confirm_live_trading():
        print("❌ Live trading activation cancelled")
        return False
    
    # Step 2: Safety checks
    if not await run_safety_checks():
        print("❌ Safety checks failed - Live trading not enabled")
        return False
    
    # Step 3: Create backup
    create_backup()
    
    # Step 4: Enable live trading
    enable_live_trading()
    
    print("\n🎉 LIVE TRADING SUCCESSFULLY ENABLED!")
    print("=" * 50)
    print("✅ Real trading is now active")
    print("✅ Paper trading is disabled")
    print("⚠️  Monitor your positions closely")
    print("⚠️  Be prepared to disable if needed")
    print("")
    print("To disable live trading, run:")
    print("  python disable_live_trading.py")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Activation cancelled by user")
        sys.exit(1)
EOF

    chmod +x activate_live_trading.py
    log "Live trading activation script created ✅"
}

# Create live trading disable script
create_disable_script() {
    log "Creating live trading disable script..."
    
    cat > disable_live_trading.py << 'EOF'
#!/usr/bin/env python3
"""
Disable Live Trading Script
Safely disables live trading and returns to paper trading mode
"""

import os
from dotenv import set_key

def disable_live_trading():
    """Disable live trading and return to paper mode"""
    env_file = '.env'
    
    # Update environment variables
    set_key(env_file, 'LIVE_TRADING_ENABLED', 'false')
    set_key(env_file, 'PAPER_TRADING_MODE', 'true')
    
    print("✅ Live trading disabled")
    print("✅ Paper trading mode enabled")
    print("🔒 System is now in safe mode")

if __name__ == "__main__":
    print("🔒 Disabling Live Trading...")
    disable_live_trading()
    print("✅ Done!")
EOF

    chmod +x disable_live_trading.py
}

# Create quick start script
create_quick_start() {
    header "CREATING QUICK START SCRIPT"
    
    cat > quick_start_trading.sh << 'EOF'
#!/bin/bash

# AI Trading Empire - Quick Start
# Rapid deployment for testing

echo "🚀 AI Trading Empire - Quick Start"
echo "=================================="

# Activate environment
source trading_env/bin/activate

echo "1. Testing exchange connections..."
python test_exchanges.py

echo -e "\n2. Testing paper trading..."
python paper_trading_test.py

echo -e "\n3. Starting trading manager..."
echo "Choose your mode:"
echo "  [P] Paper Trading (Safe - Recommended)"
echo "  [L] Live Trading (Real Money - Advanced Users Only)"
echo ""

read -p "Select mode (P/L): " -n 1 -r
echo

if [[ $REPLY =~ ^[Ll]$ ]]; then
    echo "⚠️  Activating live trading mode..."
    python activate_live_trading.py
else
    echo "📄 Using paper trading mode (safe)"
fi

echo -e "\n✅ Quick start complete!"
echo "💡 Run 'python trading_manager.py' to see status"
EOF

    chmod +x quick_start_trading.sh
    log "Quick start script created ✅"
}

# Main execution function
main() {
    header "AI TRADING EMPIRE - US LIVE TRADING SETUP (PART 2)"
    info "Setting up exchange connections and live trading capabilities"
    
    check_prerequisites
    test_imports
    create_exchange_test_script
    test_exchange_connections
    create_trading_manager
    create_paper_trading_test
    create_live_trading_activation
    create_disable_script
    create_quick_start
    
    header "SETUP COMPLETE - PART 2"
    success "Live trading system setup completed successfully! ✅"
    echo
    info "🎯 What you can do now:"
    echo "1. Test paper trading: ./paper_trading_test.py"
    echo "2. Quick start demo: ./quick_start_trading.sh"
    echo "3. Check trading status: python trading_manager.py"
    echo "4. When ready for live trading: python activate_live_trading.py"
    echo
    warn "🔒 SAFETY REMINDERS:"
    echo "• Always test with paper trading first"
    echo "• Start with small position sizes"
    echo "• Monitor trades closely"
    echo "• Never risk more than you can afford to lose"
    echo
    success "🚀 Your US-compliant AI trading system is ready!"
}

# Execute main function
main "$@"