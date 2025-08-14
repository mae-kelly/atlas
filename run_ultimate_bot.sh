#!/bin/bash

echo "🚀 ULTIMATE OKX PROFIT-MAXIMIZING BOT v2.0"
echo "=========================================="
echo ""
echo "💰 FEATURES ACTIVE:"
echo "   🔥 6 Trading Strategies Running Simultaneously"
echo "   📊 20+ Technical Indicators"
echo "   🧠 Advanced ML Sentiment Analysis"
echo "   ⚡ Dynamic Leverage (1-3x)"
echo "   🎯 Kelly Criterion Position Sizing"
echo "   🛡️  Professional Risk Management"
echo "   🔄 2-Second Update Cycles"
echo ""
echo "💡 PROFIT TARGETS:"
echo "   • $50 max position size (5x larger)"
echo "   • Multi-strategy approach"
echo "   • High-frequency scalping"
echo "   • Advanced arbitrage detection"
echo "   • Grid trading automation"
echo ""

# Activate Python environment
source crypto_env/bin/activate

# Set optimal environment variables
export TMPDIR=~/tmp
export TEMP=~/tmp
export TMP=~/tmp

# Navigate to project
cd crypto_trading

# Run the ultimate bot
echo "🚀 Launching ultimate trading engine..."
echo ""

if [ -f "./target/release/crypto_trading" ]; then
    ./target/release/crypto_trading
else
    echo "❌ Binary not found! Run build script first."
    exit 1
fi
