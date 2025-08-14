import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import asyncio
import aiohttp
import json
import sys
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import subprocess
import multiprocessing as mp

try:
    import torch.backends.mps
    MPS_AVAILABLE = torch.backends.mps.is_available()
except:
    MPS_AVAILABLE = False

class MarketPredictor(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=3, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, dim_feedforward=512),
            num_layers=2
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        transformer_out = self.transformer(attn_out)
        
        pooled = torch.mean(transformer_out, dim=1)
        
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.tanh(self.fc3(x))
        
        return x

class VolumeProfileNet(nn.Module):
    def __init__(self, input_size=100):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.fc1 = nn.Linear(256 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        return x

class MLTradingEngine:
    def __init__(self):
        self.device = self._setup_device()
        self.models = {}
        self.optimizers = {}
        self.data_cache = {}
        self.feature_extractors = {}
        self.session = None
        
    def _setup_device(self):
        if MPS_AVAILABLE:
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
            
    async def initialize_models(self, symbols: List[str]):
        for symbol in symbols:
            market_model = MarketPredictor().to(self.device)
            volume_model = VolumeProfileNet().to(self.device)
            
            self.models[f"{symbol}_market"] = market_model
            self.models[f"{symbol}_volume"] = volume_model
            
            self.optimizers[f"{symbol}_market"] = optim.AdamW(
                market_model.parameters(), lr=0.001, weight_decay=0.01
            )
            self.optimizers[f"{symbol}_volume"] = optim.AdamW(
                volume_model.parameters(), lr=0.001, weight_decay=0.01
            )
            
    async def get_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
        
    async def fetch_market_data(self, symbol: str) -> Dict:
        session = await self.get_session()
        try:
            async with session.get(f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=1m&limit=1000") as response:
                if response.status == 200:
                    data = await response.json()
                    return data
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
        return {}
        
    def extract_features(self, candle_data: List[List[str]]) -> np.ndarray:
        if not candle_data:
            return np.array([])
            
        df = pd.DataFrame(candle_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy'])
        df = df.astype({
            'open': float, 'high': float, 'low': float, 
            'close': float, 'volume': float, 'volCcy': float
        })
        
        features = []
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
        df['volume_momentum'] = df['volume'] / df['volume'].shift(10) - 1
        
        # Order flow features
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        df['mfi'] = self._calculate_mfi(df)
        
        # Microstructure features
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['volume_price_trend'] = np.where(df['close'] > df['close'].shift(1), df['volume'], -df['volume'])
        df['accumulation_distribution'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        
        feature_cols = [
            'returns', 'volatility', 'rsi', 'macd', 'bb_upper', 'bb_lower',
            'volume_sma', 'price_momentum', 'volume_momentum', 'mfi',
            'spread_proxy', 'volume_price_trend', 'accumulation_distribution'
        ]
        
        df = df.dropna()
        if len(df) > 0:
            features = df[feature_cols].values
            
        return features
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        return exp1 - exp2
        
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return sma + (std * 2), sma - (std * 2)
        
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
        return mfi.fillna(50)
        
    async def predict_market_movement(self, symbol: str, features: np.ndarray) -> float:
        if f"{symbol}_market" not in self.models or len(features) < 50:
            return 0.0
            
        model = self.models[f"{symbol}_market"]
        model.eval()
        
        with torch.no_grad():
            # Prepare sequence data
            sequence_length = 50
            if len(features) >= sequence_length:
                input_data = features[-sequence_length:]
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
                prediction = model(input_tensor)
                return float(prediction.squeeze().cpu().numpy())
                
        return 0.0
        
    async def predict_volume_profile(self, symbol: str, volume_data: np.ndarray) -> float:
        if f"{symbol}_volume" not in self.models or len(volume_data) < 100:
            return 0.0
            
        model = self.models[f"{symbol}_volume"]
        model.eval()
        
        with torch.no_grad():
            if len(volume_data) >= 100:
                input_data = volume_data[-100:]
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(0).to(self.device)
                prediction = model(input_tensor)
                return float(prediction.squeeze().cpu().numpy())
                
        return 0.0
        
    async def retrain_models(self, symbol: str, features: np.ndarray, targets: np.ndarray):
        if len(features) < 100 or len(targets) < 100:
            return
            
        model_key = f"{symbol}_market"
        if model_key not in self.models:
            return
            
        model = self.models[model_key]
        optimizer = self.optimizers[model_key]
        
        model.train()
        
        # Prepare training data
        sequence_length = 50
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
            
        if len(X) == 0:
            return
            
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.MSELoss()
        
        for epoch in range(5):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
    async def generate_signals(self, symbols: List[str]) -> Dict[str, Dict]:
        signals = {}
        
        for symbol in symbols:
            try:
                market_data = await self.fetch_market_data(symbol)
                if not market_data.get('data'):
                    continue
                    
                features = self.extract_features(market_data['data'])
                if len(features) == 0:
                    continue
                    
                price_prediction = await self.predict_market_movement(symbol, features)
                
                volume_data = np.array([float(row[5]) for row in market_data['data']])
                volume_prediction = await self.predict_volume_profile(symbol, volume_data)
                
                # Combine predictions with additional analysis
                current_price = float(market_data['data'][0][4])
                price_change = features[-1][0] if len(features) > 0 else 0
                
                confidence = min(abs(price_prediction) * 2, 1.0)
                
                signal_strength = (price_prediction * 0.7 + volume_prediction * 0.3)
                
                signals[symbol] = {
                    'signal': signal_strength,
                    'confidence': confidence,
                    'price_prediction': price_prediction,
                    'volume_prediction': volume_prediction,
                    'current_price': current_price,
                    'price_change': price_change,
                    'features': features[-1].tolist() if len(features) > 0 else []
                }
                
            except Exception as e:
                logging.error(f"Error generating signals for {symbol}: {e}")
                continue
                
        return signals
        
    async def close(self):
        if self.session:
            await self.session.close()

async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No symbols provided"}))
        return
        
    symbols = sys.argv[1].split(',')
    
    engine = MLTradingEngine()
    
    try:
        await engine.initialize_models(symbols)
        signals = await engine.generate_signals(symbols)
        print(json.dumps(signals, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
    finally:
        await engine.close()

if __name__ == "__main__":
    asyncio.run(main())
