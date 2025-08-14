#!/bin/bash

set -e

echo "Deploying production crypto trading bot..."

# Build everything
cd crypto_trading

# Install bindgen for C++ integration
cargo install bindgen-cli

# Build release version with optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

cd ..

# Setup Python ML environment with M1 acceleration
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio
pip3 install numpy pandas scikit-learn aiohttp requests beautifulsoup4 transformers

# Test ML acceleration
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
print(f'CUDA available: {torch.cuda.is_available()}')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
"

# Create systemd service for 24/7 operation
sudo tee /etc/systemd/system/crypto-trading-bot.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=Advanced Crypto Trading Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD/crypto_trading
ExecStart=$PWD/crypto_trading/target/release/crypto_trading
Restart=always
RestartSec=30
Environment=RUST_LOG=info
Environment=RUST_BACKTRACE=1
StandardOutput=journal
StandardError=journal
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Setup log rotation
sudo tee /etc/logrotate.d/crypto-trading << 'LOGROTATE_EOF'
/var/log/crypto-trading.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $USER $USER
}
LOGROTATE_EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable crypto-trading-bot

echo "Production bot deployed successfully!"
echo ""
echo "Commands:"
echo "  Start: sudo systemctl start crypto-trading-bot"
echo "  Status: sudo systemctl status crypto-trading-bot"
echo "  Logs: sudo journalctl -u crypto-trading-bot -f"
echo "  Stop: sudo systemctl stop crypto-trading-bot"
