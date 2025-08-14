#!/bin/bash

set -e

echo "Starting production crypto trading bot..."

# Check if service exists
if ! systemctl list-unit-files | grep -q crypto-trading-bot.service; then
    echo "Service not installed. Run ./deploy_production_bot.sh first"
    exit 1
fi

# Get API credentials
echo "Enter your OKX API credentials:"
read -p "API Key: " API_KEY
read -s -p "Secret Key: " SECRET_KEY
echo
read -s -p "Passphrase: " PASSPHRASE
echo

# Set environment variables
export OKX_API_KEY="$API_KEY"
export OKX_SECRET_KEY="$SECRET_KEY"
export OKX_PASSPHRASE="$PASSPHRASE"

# Update service environment
sudo tee /etc/systemd/system/crypto-trading-bot.service.d/override.conf > /dev/null << ENV_EOF
[Service]
Environment="OKX_API_KEY=$API_KEY"
Environment="OKX_SECRET_KEY=$SECRET_KEY"
Environment="OKX_PASSPHRASE=$PASSPHRASE"
Environment="RUST_LOG=info"
Environment="RUST_BACKTRACE=1"
ENV_EOF

sudo systemctl daemon-reload

# Start the service
sudo systemctl start crypto-trading-bot

echo "Bot started successfully!"
echo ""
echo "Monitor with:"
echo "  sudo systemctl status crypto-trading-bot"
echo "  sudo journalctl -u crypto-trading-bot -f"
echo ""
echo "The bot will:"
echo "  - Run 24/7 even when computer sleeps"
echo "  - Protect original $500 balance"
echo "  - Execute real trades based on ML predictions"
echo "  - Automatically restart if crashed"
echo ""
echo "To stop: sudo systemctl stop crypto-trading-bot"
