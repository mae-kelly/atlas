#!/bin/bash

echo "🤖 INTELLIGENT CRYPTO TRADING BOT"
echo "=================================="
echo ""
echo "🔐 SECURITY FEATURES:"
echo "   ✅ AES-256 credential encryption"
echo "   ✅ Memory-only storage (no disk writes)"
echo "   ✅ Secure input handling"
echo "   ✅ No API keys in environment variables"
echo ""
echo "🧠 AI FEATURES:"
echo "   ✅ Machine learning sentiment analysis"
echo "   ✅ Deep market research per token"
echo "   ✅ Continuous learning from trades"
echo "   ✅ Risk assessment algorithms"
echo ""
echo "💰 TRADING FEATURES:"
echo "   ✅ Paper trading mode (no real money)"
echo "   ✅ $10 max position size"
echo "   ✅ Real-time acceleration detection"
echo "   ✅ OKX fees and slippage calculation"
echo "   ✅ Stop loss and take profit automation"
echo ""

source crypto_env/bin/activate

redis-server --daemonize yes

ulimit -n 65536

cd crypto_trading

./target/release/crypto_trading