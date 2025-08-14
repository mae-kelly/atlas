#!/bin/bash

echo "🤖 CRYPTO TRADING BOT"
echo "===================="

cd crypto_trading

if [ ! -f "target/release/crypto_trading" ]; then
    echo "Building bot..."
    cargo build --release
fi

echo "Starting trading bot..."
echo "Balance protection: $500 minimum"
echo "Mode: Simulation (safe)"
echo ""

./target/release/crypto_trading
