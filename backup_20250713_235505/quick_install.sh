#!/bin/bash
# Quick install script for AI Trading Empire

echo "🚀 AI Trading Empire - Quick Install"
echo "===================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $MAJOR_VERSION -eq 3 ]] && [[ $MINOR_VERSION -ge 8 ]] && [[ $MINOR_VERSION -le 11 ]]; then
    echo "✅ Python $PYTHON_VERSION is compatible"
else
    echo "❌ Python $PYTHON_VERSION may have compatibility issues"
    echo "⚠️  Recommended: Python 3.8-3.11"
    echo ""
    echo "Continue anyway? (y/N)"
    read -n 1 -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "\nInstallation cancelled"
        exit 1
    fi
fi

# Create virtual environment
if [[ ! -d "trading_env" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv trading_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source trading_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install essential packages first
echo "Installing essential packages..."
pip install python-dotenv aiohttp feedparser loguru

# Install compatible requirements
if [[ -f "requirements_compatible.txt" ]]; then
    echo "Installing compatible requirements..."
    pip install -r requirements_compatible.txt
elif [[ -f "requirements.txt" ]]; then
    echo "Installing requirements (may have compatibility issues)..."
    pip install -r requirements.txt --no-deps
    pip install numpy pandas matplotlib seaborn aiohttp requests
else
    echo "Installing minimal requirements..."
    pip install numpy pandas matplotlib aiohttp requests python-dotenv loguru
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Configure API keys: cp config/api_credentials.env .env"
echo "2. Edit .env with your API keys"
echo "3. Test: python scripts/test_basic_apis.py"
echo "4. Run demo: ./run_simple_live_demo.sh"
