#!/bin/bash
echo "🚀 AI Trading Empire - Quick Start"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Installation complete!"
echo "Next: cp config/api_credentials.env .env"
echo "Then: ./run_simple_live_demo.sh"
