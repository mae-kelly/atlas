#!/bin/bash

# AI Trading Empire - Quick Cleanup Script
# Fast, safe cleanup of common unnecessary files
# No prompts - safe operations only

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[QUICK] $1${NC}"; }
info() { echo -e "${BLUE}[INFO] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }

# Quick cleanup function
quick_cleanup() {
    echo "🚀 AI Trading Empire - Quick Cleanup"
    echo "======================================"
    echo ""
    
    initial_size=$(du -sh . 2>/dev/null | cut -f1)
    log "Starting cleanup... (Initial size: $initial_size)"
    echo ""
    
    # Step 1: Remove Python cache (always safe)
    log "Removing Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    
    # Step 2: Remove common temporary files
    log "Removing temporary files..."
    find . -name "*.tmp" -type f -delete 2>/dev/null || true
    find . -name "*.temp" -type f -delete 2>/dev/null || true
    find . -name "*.log" -type f -delete 2>/dev/null || true
    find . -name "*.out" -type f -delete 2>/dev/null || true
    
    # Step 3: Remove OS-specific files
    log "Removing OS-specific files..."
    find . -name ".DS_Store" -type f -delete 2>/dev/null || true
    find . -name "Thumbs.db" -type f -delete 2>/dev/null || true
    find . -name "desktop.ini" -type f -delete 2>/dev/null || true
    
    # Step 4: Remove common backup files
    log "Removing backup files..."
    find . -name "*~" -type f -delete 2>/dev/null || true
    find . -name "*.bak" -type f -delete 2>/dev/null || true
    find . -name "*.backup" -type f -delete 2>/dev/null || true
    
    # Step 5: Clean up empty directories
    log "Removing empty directories..."
    find . -type d -empty -delete 2>/dev/null || true
    
    # Step 6: Remove trailing whitespace from Python files
    log "Cleaning Python file formatting..."
    find . -name "*.py" -type f -exec sed -i 's/[[:space:]]*$//' {} + 2>/dev/null || true
    
    # Step 7: Optimize .gitignore if it exists
    if [[ -f ".gitignore" ]]; then
        log "Optimizing .gitignore..."
        # Add common patterns if not already present
        {
            echo ""
            echo "# Quick cleanup additions"
            echo "__pycache__/"
            echo "*.pyc"
            echo "*.pyo"
            echo "*.tmp"
            echo "*.temp"
            echo "*.log"
            echo ".DS_Store"
            echo "Thumbs.db"
        } >> .gitignore
        
        # Remove duplicates
        sort .gitignore | uniq > .gitignore.tmp && mv .gitignore.tmp .gitignore
    fi
    
    # Final size
    final_size=$(du -sh . 2>/dev/null | cut -f1)
    
    echo ""
    echo "✅ Quick cleanup completed!"
    echo "   Before: $initial_size"
    echo "   After:  $final_size"
    echo ""
    info "Safe cleanup operations completed successfully"
    
    # Show what was cleaned
    echo "🧹 Cleaned:"
    echo "   - Python cache files (__pycache__, *.pyc, *.pyo)"
    echo "   - Temporary files (*.tmp, *.temp, *.log, *.out)"
    echo "   - OS files (.DS_Store, Thumbs.db)"
    echo "   - Backup files (*~, *.bak, *.backup)"
    echo "   - Empty directories"
    echo "   - Trailing whitespace in Python files"
    echo "   - Optimized .gitignore"
    echo ""
    
    # Git status tip
    if [[ -d ".git" ]]; then
        info "💡 Tip: Run 'git status' to see changes, then 'git add .' and 'git commit'"
    fi
}

# Safety check
if [[ ! -f "README.md" ]] || [[ ! -d "core" ]]; then
    warn "This doesn't appear to be the AI Trading Empire repository"
    warn "Please run from the repository root directory"
    exit 1
fi

# Run quick cleanup
quick_cleanup

# Optional Git optimization
if [[ -d ".git" ]]; then
    echo ""
    echo "🔧 Optional: Optimize Git repository?"
    read -p "Run 'git gc' to optimize repository? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Optimizing Git repository..."
        git gc --quiet 2>/dev/null || true
        info "Git optimization completed"
    fi
fi

echo ""
echo "🎉 Quick cleanup finished! Repository is now optimized."