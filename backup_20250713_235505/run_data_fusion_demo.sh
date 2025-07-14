#!/bin/bash

echo "🧬 Starting Data Fusion Engine Demo..."
echo "This demo shows how price and sentiment data combine to generate trading signals"
echo ""

# Ensure we're in the right directory
if [ ! -f "demo_data_fusion.py" ]; then
    echo "❌ demo_data_fusion.py not found. Please run from the correct directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "📦 Installing/updating required packages..."
pip install -q textblob vaderSentiment loguru numpy pandas scipy scikit-learn

# Download VADER lexicon
echo "📊 Downloading sentiment analysis data..."
python -c "
try:
    import nltk
    nltk.download('punkt', quiet=True)
    print('✅ NLTK data downloaded')
except:
    print('ℹ️  NLTK download skipped')
"

# Run the demo
echo ""
echo "🚀 Launching Data Fusion Engine Demo..."
echo "   Watch for correlations between price movements and sentiment!"
echo "   Press Ctrl+C to stop the demo early"
echo ""

python demo_data_fusion.py

echo ""
echo "✅ Demo completed!"
echo "💡 Next: Try running with real data streams for live analysis"
