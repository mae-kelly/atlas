import asyncio
import json
import websockets
import aiohttp
import time
from typing import List, Callable, Dict
from loguru import logger
from collections import deque
import numpy as np
class BinanceRealFeed:
    """
    Real-time Binance price feed with WebSocket + REST API integration
    """
    def __init__(self, symbols: List[str], on_price_update: Callable):
        self.symbols = [s.lower() for s in symbols]
        self.on_price_update = on_price_update
        self.price_history = {symbol: deque(maxlen=1000) for symbol in symbols}
        self.last_prices = {}
        self.ws_base = "wss://stream.binance.com:9443/ws"
        self.rest_base = "https://api.binance.com/api/v3"
        self.last_rest_call = 0
        self.rest_limit_delay = 0.1
    async def get_historical_data(self, symbol: str, interval: str = "1m", limit: int = 500) -> List[Dict]:
        """
        Fetch historical kline/candlestick data from Binance REST API
        """
        await self._rate_limit()
        try:
            url = f"{self.rest_base}/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        formatted_data = []
                        for kline in data:
                            formatted_data.append({
                                'symbol': symbol.upper(),
                                'timestamp': int(kline[0]) / 1000,  # Convert to seconds
                                'open': float(kline[1]),
                                'high': float(kline[2]),
                                'low': float(kline[3]),
                                'close': float(kline[4]),
                                'volume': float(kline[5]),
                                'quote_volume': float(kline[7]),
                                'trades': int(kline[8])
                            })
                        logger.info(f"📊 Fetched {len(formatted_data)} historical data points for {symbol}")
                        return formatted_data
                    else:
                        logger.error(f"❌ REST API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"❌ Historical data fetch error: {e}")
            return []
    async def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all symbols via REST API
        """
        await self._rate_limit()
        try:
            url = f"{self.rest_base}/ticker/price"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        prices = {}
                        for ticker in data:
                            symbol = ticker['symbol']
                            if symbol.lower() in self.symbols:
                                prices[symbol] = float(ticker['price'])
                        self.last_prices.update(prices)
                        return prices
                    else:
                        logger.error(f"❌ Price fetch error: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"❌ Current price fetch error: {e}")
            return {}
    async def start_websocket_stream(self):
        """
        Start real-time WebSocket price stream
        """
        stream_names = []
        for symbol in self.symbols:
            stream_names.extend([
                f"{symbol}@ticker",    # 24hr ticker statistics
                f"{symbol}@trade",     # Real-time trades
                f"{symbol}@kline_1m"   # 1-minute klines
            ])
        stream_url = f"{self.ws_base}/{'/'.join(stream_names)}"
        logger.info(f"🔗 Connecting to Binance WebSocket: {len(self.symbols)} symbols")
        while True:
            try:
                async with websockets.connect(stream_url, ping_interval=20) as ws:
                    logger.info(f"✅ Connected to Binance real-time feed")
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self._process_websocket_message(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"❌ Message processing error: {e}")
            except Exception as e:
                logger.warning(f"🔄 WebSocket disconnected: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
    async def _process_websocket_message(self, data: Dict):
        """
        Process incoming WebSocket messages
        """
        try:
            stream = data.get('stream', '')
            event_data = data.get('data', data)
            if '@ticker' in stream:
                symbol = event_data['s']
                price_data = {
                    'symbol': symbol,
                    'price': float(event_data['c']),  # Close price
                    'volume': float(event_data['v']),  # Volume
                    'price_change': float(event_data['P']),  # Price change %
                    'high': float(event_data['h']),  # 24hr high
                    'low': float(event_data['l']),   # 24hr low
                    'timestamp': time.time(),
                    'source': 'binance_ws_ticker'
                }
                self.price_history[symbol].append(price_data)
                self.last_prices[symbol] = price_data['price']
                await self.on_price_update(price_data)
            elif '@trade' in stream:
                symbol = event_data['s']
                price_data = {
                    'symbol': symbol,
                    'price': float(event_data['p']),
                    'quantity': float(event_data['q']),
                    'timestamp': int(event_data['T']) / 1000,  # Trade time
                    'is_buyer_maker': event_data['m'],
                    'source': 'binance_ws_trade'
                }
                await self.on_price_update(price_data)
            elif '@kline' in stream:
                kline = event_data['k']
                if kline['x']:  # Only process closed klines
                    symbol = kline['s']
                    price_data = {
                        'symbol': symbol,
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'timestamp': int(kline['t']) / 1000,
                        'source': 'binance_ws_kline'
                    }
                    self.price_history[symbol].append(price_data)
                    await self.on_price_update(price_data)
        except Exception as e:
            logger.error(f"❌ WebSocket message processing error: {e}")
    async def _rate_limit(self):
        """Apply rate limiting for REST API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_rest_call
        if time_since_last < self.rest_limit_delay:
            await asyncio.sleep(self.rest_limit_delay - time_since_last)
        self.last_rest_call = time.time()
    def get_price_statistics(self, symbol: str) -> Dict:
        """Get price statistics for a symbol"""
        if symbol not in self.price_history or len(self.price_history[symbol]) == 0:
            return {}
        prices = [p['price'] for p in self.price_history[symbol] if 'price' in p]
        if len(prices) < 2:
            return {}
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        return {
            'current_price': prices[-1],
            'price_change_24h': ((prices[-1] - prices[0]) / prices[0]) if len(prices) > 0 else 0,
            'volatility': float(np.std(returns)) if len(returns) > 0 else 0,
            'volume_avg': np.mean([p.get('volume', 0) for p in self.price_history[symbol]]),
            'data_points': len(self.price_history[symbol])
        }