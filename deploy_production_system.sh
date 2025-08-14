#!/bin/bash

echo "🚀 DEPLOYING PRODUCTION TRADING SYSTEM"
echo "======================================"

# Install Python dependencies
pip3 install torch torchvision torchaudio numpy pandas aiohttp requests

# Create ML script
cat > crypto_trading/ml_engine/simple_ml.py << 'MLEOF'
import sys
import json
import asyncio
import aiohttp
import numpy as np
from typing import Dict, List

class SimpleTradingML:
    async def analyze_symbols(self, symbols: List[str]) -> Dict[str, float]:
        signals = {}
        
        for symbol in symbols:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('code') == '0' and data.get('data'):
                                price_data = data['data'][0]
                                last_price = float(price_data['last'])
                                vol_24h = float(price_data.get('vol24h', 0))
                                
                                # Simple momentum signal
                                signal = np.tanh((vol_24h / 1000000) - 0.5) * 0.5
                                signals[symbol] = signal
                                
            except Exception as e:
                signals[symbol] = 0.0
                
        return signals

async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No symbols provided"}))
        return
        
    symbols = sys.argv[1].split(',')
    ml_engine = SimpleTradingML()
    
    try:
        results = await ml_engine.analyze_symbols(symbols)
        print(json.dumps(results))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    asyncio.run(main())
MLEOF

# Create systemd service
sudo tee /etc/systemd/system/crypto-trading.service > /dev/null << 'SERVICEEOF'
[Unit]
Description=Advanced Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD/crypto_trading
ExecStart=$PWD/crypto_trading/target/release/crypto_trading
Restart=always
RestartSec=30
Environment=RUST_LOG=info
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Enable service
sudo systemctl daemon-reload
sudo systemctl enable crypto-trading

echo "✅ Production system deployed!"
echo ""
echo "To start trading:"
echo "1. Set your API keys:"
echo "   export OKX_API_KEY='your_key'"
echo "   export OKX_SECRET_KEY='your_secret'" 
echo "   export OKX_PASSPHRASE='your_passphrase'"
echo ""
echo "2. Start the service:"
echo "   sudo systemctl start crypto-trading"
echo ""
echo "3. Monitor with:"
echo "   sudo systemctl status crypto-trading"
echo "   sudo journalctl -u crypto-trading -f"
