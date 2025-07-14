#!/bin/bash

echo "🌐 Starting Simple Live Data Integration Demo..."
echo "Real Binance prices + Reddit sentiment without heavy ML dependencies"
echo ""

# Activate virtual environment
source venv/bin/activate

# Install only essential packages
echo "📦 Installing essential packages..."
pip install -q aiohttp loguru numpy pandas textblob vaderSentiment

echo ""
echo "🚀 Starting 5-minute live data demo..."
echo "   📈 Real Binance prices every 3 seconds"
echo "   🐦 Live Reddit sentiment analysis"
echo "   🧬 Real-time correlation detection"
echo "   Press Ctrl+C to stop early"
echo ""

python demo_simple_live_data.py

echo ""
echo "✅ Simple Live Demo completed!"
echo "💡 Successfully integrated real market data!"
