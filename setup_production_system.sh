#!/bin/bash

echo "Setting up production trading system..."

# Install dependencies
pip3 install torch torchvision torchaudio numpy pandas aiohttp scikit-learn

# Build the C++ accelerators
cd crypto_trading
cargo build --release

# Set up Python ML environment
cd ml_engine
python3 -c "
import torch
import numpy as np
import pandas as pd
print('ML dependencies verified')
"

cd ..

# Create systemd service
sudo tee /etc/systemd/system/crypto-trading.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/crypto_trading
ExecStart=/home/$USER/crypto_trading/target/release/crypto_trading
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable crypto-trading
sudo systemctl start crypto-trading

echo "Production system setup complete"
echo "Monitor with: sudo systemctl status crypto-trading"
echo "View logs with: sudo journalctl -u crypto-trading -f"
