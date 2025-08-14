#!/bin/bash

echo "🚀 STARTING 24/7 AUTONOMOUS TRADING SYSTEM"
echo "=========================================="

# Set environment variables
export RUST_LOG=info
export RUST_BACKTRACE=1

# Ensure directories exist
mkdir -p data logs

# Build if needed
if [ ! -f "./target/release/autonomous_trader" ]; then
    echo "🔨 Building autonomous trader..."
    cargo build --release
fi

# Start the trader
echo "🤖 Launching autonomous trader..."
echo "💰 Protected Balance: $500"
echo "🛡️  Balance Protection: ACTIVE"
echo "🌙 24/7 Operation: ENABLED"
echo ""
echo "To stop: Press Ctrl+C or run 'docker stop autonomous_trader'"
echo ""

./target/release/autonomous_trader 2>&1 | tee logs/trading_$(date +%Y%m%d_%H%M%S).log
