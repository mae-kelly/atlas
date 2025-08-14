#!/bin/bash

echo "🍎 STARTING MAC NATIVE AUTONOMOUS TRADER"
echo "======================================="

# Check if setup is needed
if [ ! -d "crypto_trading" ]; then
    echo "🛠️  Running initial setup..."
    ./setup_mac_native.sh
fi

cd crypto_trading

# Check if binary exists
if [ ! -f "./target/release/crypto_trading" ]; then
    echo "🔨 Building the bot..."
    cargo build --release
fi

echo "🚀 STARTING 24/7 AUTONOMOUS TRADING SYSTEM"
echo "=========================================="
echo ""
echo "💰 Protected Balance: $500 (NEVER goes below this)"
echo "🛡️  Safety: Only trades with profits above $500"
echo "🤖 Autonomous: Runs 24/7 without supervision"
echo "🌙 Sleep Safe: You can close your laptop and it keeps running"
echo "📈 Strategies: 5 proven trading strategies running simultaneously"
echo ""
echo "🎯 PROFIT TARGETS:"
echo "   • Momentum Scalping: Quick 1.5% gains"
echo "   • Breakout Capture: 2.4% breakout trades"
echo "   • Volume Spikes: Catch unusual activity"
echo "   • Trend Following: Ride strong trends"
echo "   • Support/Resistance: Trade key levels"
echo ""
echo "⚡ FEATURES:"
echo "   • 3:1 Risk/Reward ratio on all trades"
echo "   • Maximum 2% risk per trade"
echo "   • 10% daily risk limit"
echo "   • Auto stop-loss and take-profit"
echo "   • Real-time market data from OKX"
echo ""
echo "🛑 To stop: Press Ctrl+C"
echo ""

# Run the bot with logging
exec ./target/release/crypto_trading 2>&1 | tee ../trading_log_$(date +%Y%m%d_%H%M%S).log
