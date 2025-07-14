#!/bin/bash

echo "🌐 Starting REAL Data Integration Demo..."
echo "This connects to live Binance prices and real Reddit sentiment!"
echo ""

# Check for required files
if [ ! -f "demo_real_data_integration.py" ]; then
    echo "❌ demo_real_data_integration.py not found."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install additional requirements for real data
echo "📦 Installing real data requirements..."
pip install -q aiohttp websockets textblob vaderSentiment loguru numpy pandas scipy
pip install -q xgboost lightgbm scikit-learn joblib

# Create data directories
mkdir -p data/models data/historical

echo ""
echo "⚠️  IMPORTANT: This demo uses REAL data feeds:"
echo "   📈 Binance WebSocket API (live prices)"
echo "   🐦 Reddit API (live social sentiment)"
echo "   🧬 Real-time correlation analysis"
echo "   🤖 Live ML alpha predictions"
echo ""
echo "🚀 Starting 10-minute real data demo..."
echo "   Press Ctrl+C to stop early"
echo "   Watch for live trading signals!"
echo ""

python demo_real_data_integration.py

echo ""
echo "✅ Real Data Demo completed!"
echo "💡 You just saw live crypto market analysis in action!"
