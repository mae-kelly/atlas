#!/bin/bash

echo "🤖 Starting Alpha Detection Engine Demo..."
echo "This demo shows ML models learning from price-sentiment correlations"
echo ""

# Ensure we're in the right directory
if [ ! -f "demo_alpha_detection.py" ]; then
    echo "❌ demo_alpha_detection.py not found. Please run from the correct directory."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install required ML packages
echo "📦 Installing/updating ML packages..."
pip install -q xgboost lightgbm scikit-learn joblib textblob vaderSentiment loguru numpy pandas scipy

# Create necessary directories
mkdir -p data/models

# Run the demo
echo ""
echo "🚀 Launching Alpha Detection Engine Demo..."
echo "   🧠 Watch ML models learn from market patterns!"
echo "   🎯 Look for alpha predictions with high confidence"
echo "   📊 Models retrain automatically as they gather data"
echo "   Press Ctrl+C to stop the demo early"
echo ""

python demo_alpha_detection.py

echo ""
echo "✅ Alpha Detection Demo completed!"
echo "💡 The ML models learned to predict alpha from price-sentiment patterns"
echo "📈 In production, these predictions would drive position sizing decisions"
