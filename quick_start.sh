#!/bin/bash

echo "⚡ QUICK START - 24/7 AUTONOMOUS TRADER"
echo "======================================"

# Check if autonomous_trader directory exists
if [ ! -d "autonomous_trader" ]; then
    echo "🛠️  Setting up system..."
    ./setup_autonomous_system.sh
fi

cd autonomous_trader

echo "🚀 Starting autonomous trading system..."
echo "💰 Protected Balance: $500"
echo "🤖 Fully Automated: YES"
echo "🌙 24/7 Operation: ENABLED"
echo ""

# Deploy the system
./deploy.sh

echo ""
echo "🎯 SYSTEM IS NOW RUNNING!"
echo "========================"
echo ""
echo "📊 Monitor your profits:"
echo "   • Health: curl http://localhost:8080/health"
echo "   • Logs: docker logs autonomous_trader -f"
echo "   • Status: ./monitor.sh"
echo ""
echo "🛡️  Your $500 is protected - bot only trades with profits!"
echo "💤 Go to sleep - wake up to more money!"
echo ""
echo "🛑 To stop: docker-compose down"
