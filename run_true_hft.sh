#!/bin/bash

echo "⚡ TRUE HIGH-FREQUENCY TRADING SYSTEM"
echo "===================================="
echo ""
echo "🚀 HFT SPECIFICATIONS:"
echo "   • Tick Rate: 50ms (20 times per second)"
echo "   • Monitoring: ALL OKX USDT pairs (~400+ symbols)"
echo "   • Max Positions: 20 simultaneous"
echo "   • Hold Time: 1-5 seconds per trade"
echo "   • Strategies: Momentum, Scalping, Volume, Mean Reversion"
echo ""
echo "⚡ RAPID EXECUTION FEATURES:"
echo "   • Sub-second trade execution"
echo "   • Real-time bid/ask pricing"
echo "   • Basis point precision"
echo "   • Live P&L tracking"
echo ""
echo "📊 WHAT YOU'LL SEE:"
echo "   • Constant stream of trades"
echo "   • Multiple trades per second"
echo "   • Real-time market scanning"
echo "   • Live position management"
echo ""

read -p "🚨 WARNING: This will execute RAPID paper trades. Continue? (y/N): " confirm

if [[ $confirm != [yY] ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "⚡ STARTING TRUE HFT ENGINE..."
echo ""

cd crypto_trading
./target/release/crypto_trading
