#!/bin/bash

# AI Trading Empire Cleanup Script
# This script removes unnecessary files and directories from the project structure

set -e  # Exit on any error

echo "🧹 Starting AI Trading Empire cleanup..."

# Function to safely remove files/directories
safe_remove() {
    if [ -e "$1" ]; then
        echo "Removing: $1"
        rm -rf "$1"
    else
        echo "Not found (skipping): $1"
    fi
}

# Remove version tag artifacts (these look like git artifacts)
echo "📦 Removing version artifacts..."
find . -name "=*.*.*" -type f -delete 2>/dev/null || true
find . -name "=*.*" -type f -delete 2>/dev/null || true
find . -name "=*" -type f -delete 2>/dev/null || true

# Remove backup directories
echo "🗂️ Removing backup directories..."
safe_remove "backups/mock_replacement_20250712_190646"
safe_remove "backups/mock_replacement_20250712_190928"
safe_remove "backups/size_optimization_20250712_191933"
safe_remove "backups"

# Remove duplicate package structure
echo "📁 Removing duplicate package structure..."
safe_remove "packages/ai_trading_empire_lite"
safe_remove "packages"

# Remove compressed package files
echo "📦 Removing compressed packages..."
safe_remove "ai_trading_empire_lite.tar.gz"
safe_remove "ai_trading_empire_lite.zip"

# Remove duplicate demo files (keep only the main ones)
echo "🎯 Cleaning up demo files..."
safe_remove "demo_data_fusion_final_fix.py"
safe_remove "demo_data_fusion_fixed.py"
# Keep: demo_data_fusion.py (the main one)

# Remove empty docker structure
echo "🐳 Removing empty docker directories..."
safe_remove "docker/services"
safe_remove "docker/volumes"
if [ -d "docker" ] && [ -z "$(ls -A docker)" ]; then
    safe_remove "docker"
fi

# Remove fallbacks directory (should be integrated into main modules)
echo "🔄 Removing fallbacks directory..."
safe_remove "fallbacks"

# Remove experimental/questionable ML components
echo "🧠 Removing experimental ML components..."
safe_remove "ml/models/gnn_advanced/quantum_nets.py"
safe_remove "infrastructure/fpga"

# Remove duplicate .env files (keep main one)
echo "⚙️ Cleaning up environment files..."
safe_remove "config/api_credentials.env"
safe_remove "packages/ai_trading_empire_lite/config/api_credentials.env"
# Keep: .env in root

# Remove backup requirement files
echo "📋 Removing backup requirements..."
safe_remove "size_optimization_20250712_191933/requirements.txt.backup"

# Remove empty __init__.py files in directories with no other Python files
echo "🐍 Cleaning up empty __init__.py files..."
find . -name "__init__.py" -type f | while read -r init_file; do
    dir_path=$(dirname "$init_file")
    # Count Python files (excluding __init__.py) in the directory
    py_count=$(find "$dir_path" -maxdepth 1 -name "*.py" ! -name "__init__.py" | wc -l)
    # Count subdirectories
    subdir_count=$(find "$dir_path" -maxdepth 1 -type d ! -path "$dir_path" | wc -l)
    
    # If no Python files and no subdirectories, remove the __init__.py
    if [ "$py_count" -eq 0 ] && [ "$subdir_count" -eq 0 ]; then
        echo "Removing empty __init__.py: $init_file"
        rm -f "$init_file"
    fi
done

# Remove empty directories
echo "📂 Removing empty directories..."
find . -type d -empty -delete 2>/dev/null || true

# Clean up redundant shell scripts (keep essential ones)
echo "🔧 Cleaning up shell scripts..."
safe_remove "compress_large_files.sh"
safe_remove "decompress_files.sh"
safe_remove "optimize_environment.sh"
safe_remove "restore_full_environment.sh"
# Keep: one.sh, two.sh, quick_install.sh, and run_*_demo.sh scripts

# Remove redundant test files
echo "🧪 Cleaning up test files..."
safe_remove "test_compatibility.py"
# Keep other test files as they seem to test different components

# Remove setup artifacts
echo "🔨 Removing setup artifacts..."
safe_remove "setup.log"

# Remove API status report (should be generated, not stored)
echo "📊 Removing generated reports..."
safe_remove "api_status_report.json"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📈 Summary of what was removed:"
echo "   • Version tag artifacts"
echo "   • Backup directories"
echo "   • Duplicate package structure"
echo "   • Compressed package files"
echo "   • Duplicate demo files"
echo "   • Empty docker directories"
echo "   • Fallbacks directory"
echo "   • Experimental quantum ML components"
echo "   • FPGA infrastructure"
echo "   • Duplicate environment files"
echo "   • Empty __init__.py files"
echo "   • Empty directories"
echo "   • Redundant shell scripts"
echo "   • Setup artifacts"
echo ""
echo "🎉 Your AI trading empire is now leaner and cleaner!"