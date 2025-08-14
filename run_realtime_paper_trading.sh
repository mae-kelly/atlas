#!/bin/bash

echo "🔴 REAL-TIME PAPER TRADING SYSTEM"
echo "================================="
echo ""
echo "📊 LIVE FEED FEATURES:"
echo "   • 200ms price updates (5x per second)"
echo "   • 100ms signal processing (10x per second)"
echo "   • 500ms display updates (2x per second)"
echo "   • Real OKX prices and spreads"
echo "   • Paper trading with real fees"
echo ""
echo "💰 PAPER TRADING SETTINGS:"
echo "   • $500 starting balance"
echo "   • $10-$50 position sizes"
echo "   • Real execution simulation"
echo "   • Live P&L tracking"
echo ""
echo "⚡ SPEED FEATURES:"
echo "   • Velocity tracking (%/second)"
echo "   • 1s, 5s, 30s price changes"
echo "   • Real-time spread monitoring"
echo "   • Instant signal detection"
echo ""

read -p "Press Enter to start real-time feed..."

cd crypto_trading
./target/release/crypto_trading
