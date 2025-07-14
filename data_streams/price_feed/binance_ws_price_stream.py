import asyncio
import json
import websockets
import aiohttp
from loguru import logger
from typing import List, Callable, Dict, Optional
import os
from dotenv import load_dotenv
import time
import hmac
import hashlib
import base64
load_dotenv()
BINANCE_WS_ENDPOINT = "wss://stream.binance.com:9443/ws"
BINANCE_API_ENDPOINT = "https://api.binance.com"
class BinancePriceStream:
    def __init__(self, pairs: List[str], on_message: Callable[[dict], None]):
        """
        Enhanced Binance WebSocket with real API integration
        """
        self.pairs = [p.lower() for p in pairs]
        self.on_message = on_message
        self.ws_url = self._construct_url()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.rate_limiter = asyncio.Semaphore(1200)
        self.reconnect_attempts = 0
        self.max_reconnects = 10
        self.last_heartbeat = time.time()
        self.last_prices = {}
        self.price_change_threshold = 0.1
    def _construct_url(self) -> str:
        """Construct WebSocket URL with multiple streams"""
        streams = "/".join([f"{pair}@ticker" for pair in self.pairs])
        return f"{BINANCE_WS_ENDPOINT}/{streams}"
    async def connect(self):
        """Enhanced connection with reconnection logic"""
        while self.reconnect_attempts < self.max_reconnects:
            try:
                logger.info(f"🟢 Connecting to Binance WebSocket (attempt {self.reconnect_attempts + 1})")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as ws:
                    logger.info(f"✅ Connected to Binance for: {', '.join(self.pairs)}")
                    self.reconnect_attempts = 0
                    await self._subscribe_to_streams(ws)
                    async for message in ws:
                        try:
                            await self._process_message(message)
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"❌ Message processing error: {e}")
            except websockets.exceptions.ConnectionClosed:
                logger.warning("🔌 WebSocket connection closed")
                await self._handle_reconnection()
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                await self._handle_reconnection()
    async def _subscribe_to_streams(self, ws):
        """Subscribe to multiple data streams"""
        subscription = {
            "method": "SUBSCRIBE",
            "params": [f"{pair}@ticker" for pair in self.pairs],
            "id": 1
        }
        await ws.send(json.dumps(subscription))
        trade_subscription = {
            "method": "SUBSCRIBE", 
            "params": [f"{pair}@trade" for pair in self.pairs],
            "id": 2
        }
        await ws.send(json.dumps(trade_subscription))
    async def _process_message(self, message: str):
        """Enhanced message processing with validation"""
        data = json.loads(message)
        if 'stream' in data:
            stream_data = data['data']
            stream_name = data['stream']
            if '@ticker' in stream_name:
                await self._process_ticker_data(stream_data)
            elif '@trade' in stream_name:
                await self._process_trade_data(stream_data)
        elif 'e' in data:  # Event data
            if data['e'] == '24hrTicker':
                await self._process_ticker_data(data)
            elif data['e'] == 'trade':
                await self._process_trade_data(data)
    async def _process_ticker_data(self, data: dict):
        """Process 24hr ticker statistics"""
        try:
            symbol = data.get('s', '').upper()
            processed_data = {
                'symbol': symbol,
                'price': float(data.get('c', 0)),  # Current price
                'volume': float(data.get('v', 0)),  # 24hr volume
                'high': float(data.get('h', 0)),    # 24hr high
                'low': float(data.get('l', 0)),     # 24hr low
                'open': float(data.get('o', 0)),    # 24hr open
                'change': float(data.get('P', 0)),  # 24hr price change %
                'timestamp': int(data.get('E', time.time() * 1000)) / 1000,
                'source': 'binance_ticker',
                'data_type': 'ticker'
            }
            if await self._validate_price_data(processed_data):
                await asyncio.create_task(self.on_message(processed_data))
        except Exception as e:
            logger.error(f"❌ Ticker data processing error: {e}")
    async def _process_trade_data(self, data: dict):
        """Process individual trade data"""
        try:
            symbol = data.get('s', '').upper()
            processed_data = {
                'symbol': symbol,
                'price': float(data.get('p', 0)),      # Trade price
                'quantity': float(data.get('q', 0)),   # Trade quantity
                'timestamp': int(data.get('T', time.time() * 1000)) / 1000,
                'is_buyer_maker': data.get('m', False),
                'trade_id': data.get('t', 0),
                'source': 'binance_trade',
                'data_type': 'trade'
            }
            if await self._validate_price_data(processed_data):
                await asyncio.create_task(self.on_message(processed_data))
        except Exception as e:
            logger.error(f"❌ Trade data processing error: {e}")
    async def _validate_price_data(self, data: dict) -> bool:
        """Validate incoming price data for anomalies"""
        symbol = data['symbol']
        current_price = data['price']
        if current_price <= 0:
            logger.warning(f"⚠️ Invalid price for {symbol}: {current_price}")
            return False
        if symbol in self.last_prices:
            last_price = self.last_prices[symbol]
            price_change = abs(current_price - last_price) / last_price
            if price_change > self.price_change_threshold:
                logger.warning(f"🚨 Large price movement detected for {symbol}: "
                             f"{last_price} -> {current_price} ({price_change:.2%})")
        self.last_prices[symbol] = current_price
        return True
    async def _handle_reconnection(self):
        """Handle reconnection with exponential backoff"""
        self.reconnect_attempts += 1
        wait_time = min(300, 2 ** self.reconnect_attempts)
        logger.warning(f"🔄 Reconnecting in {wait_time} seconds (attempt {self.reconnect_attempts})")
        await asyncio.sleep(wait_time)
    async def get_historical_klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[Dict]:
        """Fetch historical candlestick data via REST API"""
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    url = f"{BINANCE_API_ENDPOINT}/api/v3/klines"
                    params = {
                        'symbol': symbol.upper(),
                        'interval': interval,
                        'limit': limit
                    }
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            klines = []
                            for kline in data:
                                klines.append({
                                    'timestamp': int(kline[0]) / 1000,
                                    'open': float(kline[1]),
                                    'high': float(kline[2]),
                                    'low': float(kline[3]),
                                    'close': float(kline[4]),
                                    'volume': float(kline[5]),
                                    'symbol': symbol.upper()
                                })
                            return klines
                        else:
                            logger.error(f"❌ Binance API error: {response.status}")
                            return []
        except Exception as e:
            logger.error(f"❌ Historical data fetch error: {e}")
            return []
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Fetch current order book"""
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    url = f"{BINANCE_API_ENDPOINT}/api/v3/depth"
                    params = {
                        'symbol': symbol.upper(),
                        'limit': limit
                    }
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                'symbol': symbol.upper(),
                                'bids': [[float(bid[0]), float(bid[1])] for bid in data['bids']],
                                'asks': [[float(ask[0]), float(ask[1])] for ask in data['asks']],
                                'timestamp': time.time()
                            }
                        else:
                            logger.error(f"❌ Order book API error: {response.status}")
                            return {}
        except Exception as e:
            logger.error(f"❌ Order book fetch error: {e}")
            return {}
class EnhancedBinanceDataCollector:
    """Comprehensive Binance data collection with multiple endpoints"""
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.session = None
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    def _generate_signature(self, params: dict) -> str:
        """Generate API signature for authenticated requests"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.secret_key.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
    async def get_account_info(self) -> Dict:
        """Get account information (requires API key)"""
        if not self.api_key or not self.secret_key:
            logger.warning("⚠️ Binance API credentials not provided")
            return {}
        try:
            timestamp = int(time.time() * 1000)
            params = {'timestamp': timestamp}
            signature = self._generate_signature(params)
            params['signature'] = signature
            headers = {'X-MBX-APIKEY': self.api_key}
            url = f"{BINANCE_API_ENDPOINT}/api/v3/account"
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"❌ Account info error: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"❌ Account info fetch error: {e}")
            return {}
    async def get_all_symbols(self) -> List[str]:
        """Get all trading symbols"""
        try:
            url = f"{BINANCE_API_ENDPOINT}/api/v3/exchangeInfo"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
                    return symbols
                else:
                    logger.error(f"❌ Exchange info error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"❌ Symbols fetch error: {e}")
            return []