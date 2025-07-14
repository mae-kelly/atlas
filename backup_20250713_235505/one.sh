#!/bin/bash

# AI Trading Empire - Cleanup and Compression Script
# This script removes unnecessary files and compresses the codebase

set -e

echo "🧹 AI Trading Empire - Cleanup and Compression Script"
echo "===================================================="

# Create backup directory with timestamp
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo "📦 Creating backup in: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup critical files before cleanup
echo "💾 Backing up critical files..."
cp -r config/ "$BACKUP_DIR/" 2>/dev/null || true
cp -r core/ "$BACKUP_DIR/" 2>/dev/null || true
cp *.py "$BACKUP_DIR/" 2>/dev/null || true
cp *.sh "$BACKUP_DIR/" 2>/dev/null || true
cp requirements.txt "$BACKUP_DIR/" 2>/dev/null || true

echo "✅ Backup completed"

# Function to remove files/directories safely
safe_remove() {
    if [ -e "$1" ]; then
        echo "🗑️  Removing: $1"
        rm -rf "$1"
    fi
}

# Function to compress Python files
compress_python() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "📦 Compressing: $file"
        # Remove comments, empty lines, and extra whitespace
        python3 -c "
import re
import sys

def compress_python(content):
    # Remove comments (but keep docstrings)
    lines = content.split('\n')
    compressed_lines = []
    in_docstring = False
    docstring_quote = None
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
            
        # Handle docstrings
        if ('\"\"\"' in stripped or \"'''\" in stripped):
            if not in_docstring:
                in_docstring = True
                docstring_quote = '\"\"\"' if '\"\"\"' in stripped else \"'''\"
                compressed_lines.append(line)
                if stripped.count(docstring_quote) >= 2:
                    in_docstring = False
                continue
            else:
                compressed_lines.append(line)
                if docstring_quote in stripped:
                    in_docstring = False
                continue
        
        if in_docstring:
            compressed_lines.append(line)
            continue
            
        # Remove inline comments (but not in strings)
        if '#' in stripped and not stripped.startswith('#'):
            # Simple check - avoid removing # inside strings
            if not ('\"' in stripped or \"'\" in stripped):
                line = line.split('#')[0].rstrip()
            
        # Skip comment-only lines
        if stripped.startswith('#'):
            continue
            
        compressed_lines.append(line)
    
    return '\n'.join(compressed_lines)

try:
    with open('$file', 'r') as f:
        content = f.read()
    
    compressed = compress_python(content)
    
    with open('$file', 'w') as f:
        f.write(compressed)
        
    print(f'Compressed: $file')
except Exception as e:
    print(f'Error compressing $file: {e}')
" 2>/dev/null || echo "⚠️  Could not compress $file"
    fi
}

echo ""
echo "🧹 Removing unnecessary files and directories..."

# Remove development and testing files
safe_remove "__pycache__"
safe_remove ".pytest_cache"
safe_remove "*.pyc"
safe_remove "*.pyo"
safe_remove "*.pyd"
safe_remove ".coverage"
safe_remove "htmlcov/"
safe_remove ".tox/"
safe_remove ".nox/"
safe_remove "coverage.xml"
safe_remove "*.cover"
safe_remove ".hypothesis/"

# Remove IDE and editor files
safe_remove ".vscode/"
safe_remove ".idea/"
safe_remove "*.swp"
safe_remove "*.swo"
safe_remove "*~"
safe_remove ".DS_Store"
safe_remove "Thumbs.db"

# Remove documentation files (keep essential README)
safe_remove "docs/"
safe_remove "*.md"
echo "📝 Keeping essential documentation..."
# Restore critical documentation
echo "# AI Trading Empire - Compressed Version

A high-performance cryptocurrency trading system with real-time data fusion.

## Quick Start
1. \`./quick_install.sh\`
2. \`cp config/api_credentials.env .env\`
3. Edit .env with your API keys
4. \`./run_simple_live_demo.sh\`

## Key Components
- Data Fusion Engine: Correlates price and sentiment
- Alpha Detection: ML models for prediction
- Risk Management: Kelly criterion position sizing
- Portfolio Management: Real-time tracking

## Demos
- \`./run_data_fusion_demo.sh\` - Core fusion engine
- \`./run_alpha_detection_demo.sh\` - ML predictions  
- \`./run_real_data_demo.sh\` - Live market data

## Requirements
- Python 3.8-3.11
- API keys for data sources (optional)
- See requirements.txt for dependencies

Compressed on $(date)
" > README.md

# Remove redundant demo files (keep essential ones)
safe_remove "demo_real_data_integration.py"
safe_remove "real_world_integration.py"

# Remove some advanced/experimental features
safe_remove "infrastructure/"
safe_remove "orchestration/"
safe_remove "strategies/mev/"
safe_remove "strategies/flashloan/"
safe_remove "strategies/options/"

# Remove some redundant ML models
safe_remove "ml/models/gnn/"
safe_remove "ml/models/rl/"
safe_remove "ml/models/helformer/"
safe_remove "ml/models/hybrid/"

# Clean up data directories
safe_remove "data/"
mkdir -p data/{models,historical,performance}

# Remove some API clients (keep essential ones)
safe_remove "api_clients/news_sentiment_client.py"

# Compress remaining Python files
echo ""
echo "📦 Compressing Python files..."

# Find and compress all Python files
find . -name "*.py" -type f | while read -r file; do
    # Skip __init__.py files and very small files
    if [[ ! "$file" =~ __init__.py$ ]] && [[ $(wc -l < "$file") -gt 10 ]]; then
        compress_python "$file"
    fi
done

echo ""
echo "🔧 Creating optimized requirements.txt..."

# Create minimal requirements file
cat > requirements_minimal.txt << 'EOF'
# Core Dependencies
numpy>=1.24.0
pandas>=2.0.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
loguru>=0.7.0

# API clients
requests>=2.31.0
yfinance>=0.2.18
ccxt>=4.1.0

# Sentiment Analysis
textblob>=0.17.1
vaderSentiment>=3.3.2

# Scientific Computing
scipy>=1.10.0
scikit-learn>=1.3.0

# Optional ML (for alpha detection)
xgboost>=2.0.0
lightgbm>=4.0.0

# Data feeds
feedparser>=6.0.10
websockets>=12.0
EOF

# Backup original requirements and replace
if [ -f "requirements.txt" ]; then
    mv requirements.txt "$BACKUP_DIR/requirements_original.txt"
fi
mv requirements_minimal.txt requirements.txt

echo ""
echo "🗜️  Creating compressed installation script..."

# Create ultra-minimal install script
cat > quick_start.sh << 'EOF'
#!/bin/bash
echo "🚀 AI Trading Empire - Quick Start"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Installation complete!"
echo "Next: cp config/api_credentials.env .env"
echo "Then: ./run_simple_live_demo.sh"
EOF

chmod +x quick_start.sh

echo ""
echo "📊 Analyzing space savings..."

# Calculate size before/after
if [ -d "$BACKUP_DIR" ]; then
    BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "Unknown")
    CURRENT_SIZE=$(du -sh . --exclude="$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "Unknown")
    
    echo "📈 Space Analysis:"
    echo "   Original size: $BACKUP_SIZE"
    echo "   Compressed size: $CURRENT_SIZE"
fi

echo ""
echo "🧹 Cleanup Summary:"
echo "   ✅ Removed development files"
echo "   ✅ Removed redundant demos"
echo "   ✅ Removed experimental features"
echo "   ✅ Compressed Python code"
echo "   ✅ Minimized dependencies"
echo "   ✅ Created quick start script"

echo ""
echo "📁 Remaining Structure:"
find . -maxdepth 2 -type d ! -path "./.*" ! -path "./$BACKUP_DIR*" | sort

echo ""
echo "✨ Compression Complete!"
echo ""
echo "🚀 To start using the compressed system:"
echo "   1. ./quick_start.sh"
echo "   2. cp config/api_credentials.env .env"  
echo "   3. Edit .env with your API keys"
echo "   4. ./run_simple_live_demo.sh"
echo ""
echo "💾 Backup saved in: $BACKUP_DIR"
echo "⚠️  Keep the backup until you verify everything works!"

# Make demo scripts executable
chmod +x *.sh 2>/dev/null || true

echo ""
echo "🎯 Compressed AI Trading Empire is ready for deployment!"