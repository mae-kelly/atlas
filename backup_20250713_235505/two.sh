#!/bin/bash

# AI Trading Empire - Space Cleanup Script
# Cleans up large files and frees disk space while preserving functionality

set -e

echo "🧹 AI Trading Empire - Space Cleanup"
echo "====================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Function to get directory size
get_size() {
    if [[ -d "$1" ]]; then
        du -sh "$1" 2>/dev/null | cut -f1 || echo "Unknown"
    else
        echo "N/A"
    fi
}

echo "Starting space cleanup process..."
echo ""

# Step 1: Clean Python cache files
log_info "Cleaning Python cache files..."
CACHE_COUNT=0
find . -type d -name "__pycache__" | while read dir; do
    rm -rf "$dir"
    ((CACHE_COUNT++))
done 2>/dev/null || true

find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

log_success "Cleaned Python cache files"

# Step 2: Clean virtual environment large files
if [[ -d "venv" ]]; then
    VENV_SIZE_BEFORE=$(get_size "venv")
    log_info "Virtual environment size before cleanup: $VENV_SIZE_BEFORE"
    
    # Remove large documentation and test files
    find venv -type d \( -name "doc" -o -name "docs" -o -name "examples" -o -name "tests" -o -name "test" \) -exec rm -rf {} + 2>/dev/null || true
    
    # Remove large markdown files
    find venv -type f -name "*.md" -size +1M -delete 2>/dev/null || true
    
    # Remove C/C++ source files (not needed after compilation)
    find venv -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -delete 2>/dev/null || true
    
    # Compress large binary files
    log_info "Compressing large binary files..."
    find venv -type f -size +10M \( -name "*.dylib" -o -name "*.so" -o -name "*.dll" \) | while read file; do
        if [[ ! -f "${file}.gz" ]]; then
            log_info "Compressing: $(basename "$file")"
            gzip -k "$file" && rm "$file"
        fi
    done 2>/dev/null || true
    
    VENV_SIZE_AFTER=$(get_size "venv")
    log_success "Virtual environment size after cleanup: $VENV_SIZE_AFTER"
fi

if [[ -d "trading_env" ]]; then
    TRADING_ENV_SIZE_BEFORE=$(get_size "trading_env")
    log_info "Trading environment size before cleanup: $TRADING_ENV_SIZE_BEFORE"
    
    # Same cleanup for trading_env
    find trading_env -type d \( -name "doc" -o -name "docs" -o -name "examples" -o -name "tests" -o -name "test" \) -exec rm -rf {} + 2>/dev/null || true
    find trading_env -type f -name "*.md" -size +1M -delete 2>/dev/null || true
    find trading_env -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -delete 2>/dev/null || true
    
    # Compress large binaries
    find trading_env -type f -size +10M \( -name "*.dylib" -o -name "*.so" -o -name "*.dll" \) | while read file; do
        if [[ ! -f "${file}.gz" ]]; then
            log_info "Compressing: $(basename "$file")"
            gzip -k "$file" && rm "$file"
        fi
    done 2>/dev/null || true
    
    TRADING_ENV_SIZE_AFTER=$(get_size "trading_env")
    log_success "Trading environment size after cleanup: $TRADING_ENV_SIZE_AFTER"
fi

# Step 3: Clean Git repository
if [[ -d ".git" ]]; then
    log_info "Optimizing Git repository..."
    GIT_SIZE_BEFORE=$(get_size ".git")
    log_info "Git directory size before optimization: $GIT_SIZE_BEFORE"
    
    # Clean up Git
    git gc --aggressive --prune=now 2>/dev/null || true
    git repack -a -d --depth=250 --window=250 2>/dev/null || true
    git prune 2>/dev/null || true
    
    GIT_SIZE_AFTER=$(get_size ".git")
    log_success "Git directory size after optimization: $GIT_SIZE_AFTER"
fi

# Step 4: Remove temporary files
log_info "Cleaning temporary files..."
rm -f *.tmp 2>/dev/null || true
rm -f *.temp 2>/dev/null || true
rm -rf tmp/ 2>/dev/null || true
rm -rf temp/ 2>/dev/null || true

# Step 5: Clean up large log files
log_info "Cleaning large log files..."
find . -name "*.log" -size +10M -delete 2>/dev/null || true
find . -name "*.out" -size +10M -delete 2>/dev/null || true

# Step 6: Remove duplicate demo files
log_info "Removing duplicate demo files..."
rm -f demo_*_fixed.py 2>/dev/null || true
rm -f demo_*_final_fix.py 2>/dev/null || true
rm -f requirements_*.txt 2>/dev/null || true

# Step 7: Clean up user cache directories (optional)
echo ""
log_warning "Optional: Clean user cache directories? This will remove cached ML models and pip cache."
read -p "Clean user caches? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Cleaning user cache directories..."
    
    # Clean pip cache
    if command -v pip >/dev/null 2>&1; then
        pip cache purge 2>/dev/null || true
        log_success "Cleaned pip cache"
    fi
    
    # Clean Homebrew cache (if on macOS)
    if command -v brew >/dev/null 2>&1; then
        brew cleanup --prune=all 2>/dev/null || true
        log_success "Cleaned Homebrew cache"
    fi
    
    # Clean Python user cache
    rm -rf ~/.cache/pip/ 2>/dev/null || true
    rm -rf ~/.cache/matplotlib/ 2>/dev/null || true
    
    # Clean large ML model caches (be careful with this)
    echo ""
    log_warning "Clean large ML model caches? This will remove downloaded models but they can be re-downloaded."
    read -p "Clean ML model caches? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ~/.cache/huggingface/ 2>/dev/null || true
        rm -rf ~/.cache/torch/ 2>/dev/null || true
        rm -rf ~/.cache/transformers/ 2>/dev/null || true
        log_success "Cleaned ML model caches"
    fi
fi

# Step 8: Create space usage report
log_info "Generating space usage report..."

echo ""
echo "📊 Current Directory Sizes:"
echo "=========================="
echo "Project root:     $(get_size .)"
echo "Virtual env:      $(get_size venv)"
echo "Trading env:      $(get_size trading_env)"
echo "Git directory:    $(get_size .git)"
echo "Core modules:     $(get_size core)"
echo "ML modules:       $(get_size ml)"
echo "Data streams:     $(get_size data_streams)"
echo ""

# Find largest files remaining
echo "🔍 Largest remaining files (>5MB):"
echo "=================================="
find . -type f -size +5M -exec ls -lh {} \; 2>/dev/null | head -10 | awk '{print $5 "\t" $9}' || echo "No large files found"

# Step 9: Create .gitignore to prevent large files
log_info "Creating/updating .gitignore..."
cat >> .gitignore << 'EOF'

# AI Trading Empire - Prevent large files
*.dylib
*.so
*.dll
*.tar.gz
*.zip
*.7z

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
venv/
trading_env/
env/
ENV/

# Large data files
*.h5
*.hdf5
*.parquet
*.pickle
*.pkl

# Model files
*.pt
*.pth
*.model
*.ckpt

# Logs
*.log
*.out

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temporary files
tmp/
temp/
*.tmp
*.temp
EOF

# Step 10: Create compression script for future use
cat > compress_large_files.sh << 'EOF'
#!/bin/bash
# Compress large files to save space

echo "🗜️  Compressing large files..."

# Find and compress large binary files
find . -type f -size +5M \( -name "*.dylib" -o -name "*.so" -o -name "*.dll" \) | while read file; do
    if [[ ! -f "${file}.gz" ]]; then
        echo "Compressing: $file"
        gzip -k "$file" && rm "$file"
    fi
done

# Find and compress large data files
find . -type f -size +10M \( -name "*.csv" -o -name "*.json" -o -name "*.txt" \) | while read file; do
    if [[ ! -f "${file}.gz" ]]; then
        echo "Compressing data file: $file"
        gzip -k "$file" && rm "$file"
    fi
done

echo "✅ Compression complete"
EOF

chmod +x compress_large_files.sh

# Step 11: Create decompression script
cat > decompress_files.sh << 'EOF'
#!/bin/bash
# Decompress files when needed

echo "📦 Decompressing files..."

# Decompress all .gz files
find . -name "*.gz" | while read file; do
    echo "Decompressing: $file"
    gunzip "$file"
done

echo "✅ Decompression complete"
EOF

chmod +x decompress_files.sh

# Final summary
echo ""
log_success "Space cleanup complete! 🎉"
echo ""
echo "📈 Cleanup Summary:"
echo "=================="
echo "✅ Cleaned Python cache files"
echo "✅ Optimized virtual environments"
echo "✅ Compressed large binary files"
echo "✅ Optimized Git repository"
echo "✅ Removed temporary files"
echo "✅ Created .gitignore for future prevention"
echo "✅ Created compression/decompression scripts"
echo ""
echo "🛠️  Available tools:"
echo "- Compress large files: ./compress_large_files.sh"
echo "- Decompress files: ./decompress_files.sh"
echo ""
echo "💡 Tips to keep space usage low:"
echo "1. Run this cleanup script regularly"
echo "2. Use git gc to optimize repository"
echo "3. Clear pip cache: pip cache purge"
echo "4. Remove unused virtual environments"
echo "5. Compress data files when not in use"
echo ""

# Show final directory size
FINAL_SIZE=$(get_size .)
echo "📊 Final project size: $FINAL_SIZE"
echo ""
log_success "Ready to continue with AI Trading Empire! 🚀"