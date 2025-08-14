#!/bin/bash

echo "🤖 CRYPTO TRADING BOT"
echo "===================="
echo ""

# Set environment variables
export TMPDIR=~/tmp
export TEMP=~/tmp
export TMP=~/tmp

# Activate Python environment
if [ -f "crypto_env/bin/activate" ]; then
    source crypto_env/bin/activate
    echo "✅ Python environment activated"
else
    echo "❌ Python environment not found!"
    exit 1
fi

# Navigate to project
cd crypto_trading

# Determine which binary to run
if [ -f "./target/release/crypto_trading" ]; then
    echo "🚀 Running release build..."
    BINARY="./target/release/crypto_trading"
elif [ -f "./target/debug/crypto_trading" ]; then
    echo "🚀 Running debug build..."
    BINARY="./target/debug/crypto_trading"
else
    echo "❌ No binary found! Run setup first."
    exit 1
fi

echo "💡 When prompted for credentials, you can enter:"
echo "   • API Key: demo"
echo "   • Secret Key: demo"
echo "   • Passphrase: demo"
echo ""
echo "⚠️  This is PAPER TRADING - No real money at risk!"
echo ""

# Run the bot
$BINARY
