import asyncio
import ccxt.async_support as ccxt
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time
import os
from dotenv import load_dotenv
import hmac
import hashlib
import json
import websockets
from dataclasses import dataclass

load_dotenv()

@dataclass
class OrderBookLevel:
    price: float
    quantity: float
    
@dataclass 
class OrderBook:
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel] 
    timestamp: float

class ExchangeConnector(ABC):
    """Enhanced base class for exchange connectors with real API integration"""
    
    def __init__(self, name: str, api_key: str = None, secret: str = None, sandbox: bool = True):
        self.name = name
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox
        self.exchange = None
        self.connected = False
        
        # Enhanced rate limiting
        self.rate_limits = {
            'order': 10,      # 10 orders per second
            'market': 100,    # 100 market data requests per second  
            'account': 5      # 5 account requests per second
        }
        self.rate_limit_windows = {endpoint: [] for endpoint in self.rate_limits}
        
        # Connection retry settings
        self.max_retries = 5
        self.retry_delay = 5
        self.heartbeat_interval = 30
        
        # WebSocket connections
        self.ws_connections = {}
        self.ws_subscriptions = set()
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.avg_latency = 0.0
        
    async def _check_rate_limit(self, endpoint_type: str) -> bool:
        """Enhanced rate limiting with sliding window"""
        current_time = time.time()
        window = self.rate_limit_windows[endpoint_type]
        
        # Remove old requests outside 1-second window
        window[:] = [t for t in window if current_time - t < 1.0]
        
        # Check if under limit
        if len(window) >= self.rate_limits[endpoint_type]:
            await asyncio.sleep(0.1)  # Small delay if at limit
            return False
            
        window.append(current_time)
        return True
    
    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                
                # Update performance metrics
                latency = time.time() - start_time
                self.avg_latency = (self.avg_latency * self.request_count + latency) / (self.request_count + 1)
                self.request_count += 1
                
                return result
                
            except Exception as e:
                self.error_count += 1
                
                if attempt == self.max_retries - 1:
                    raise e
                    
                wait_time = (2 ** attempt) * self.retry_delay
                logger.warning(f"⚠️ {self.name} request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    @abstractmethod
    async def connect(self):
        """Connect to exchange"""
        pass
    
    @abstractmethod 
    async def disconnect(self):
        """Disconnect from exchange"""
        pass
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker with rate limiting"""
        await self._check_rate_limit('market')
        return await self._execute_with_retry(self._fetch_ticker_impl, symbol)
    
    async def _fetch_ticker_impl(self, symbol: str) -> Dict:
        """Implementation-specific ticker fetch"""
        return await self.exchange.fetch_ticker(symbol)

class BinanceConnector(ExchangeConnector):
    """Enhanced Binance connector with full API integration"""
    
    def __init__(self, api_key: str = None, secret: str = None, sandbox: bool = True):
        super().__init__("binance", api_key, secret, sandbox)
        
        # Binance-specific settings
        self.base_url = "https://testnet.binance.vision" if sandbox else "https://api.binance.com"
        self.ws_url = "wss://testnet.binance.vision/ws" if sandbox else "wss://stream.binance.com:9443/ws"
        
        # API credentials from environment
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.secret = secret or os.getenv('BINANCE_SECRET_KEY')
        
        # Binance-specific rate limits (per minute)
        self.weight_limit = 1200
        self.current_weight = 0
        self.weight_reset_time = time.time() + 60
        
    async def connect(self):
        """Enhanced Binance connection with WebSocket support"""
        try:
            # Initialize REST API
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
            
            # Test connection and sync time
            await self._sync_time()
            await self.exchange.load_markets()
            
            # Test authentication if credentials provided
            if self.api_key and self.secret:
                account_info = await self.exchange.fetch_balance()
                logger.info(f"✅ Authenticated with Binance - Account assets: {len(account_info.get('info', {}).get('balances', []))}")
            
            self.connected = True
            logger.info(f"✅ Connected to Binance ({'Testnet' if self.sandbox else 'Mainnet'})")
            
            # Start WebSocket connections
            asyncio.create_task(self._maintain_websocket_connection())
            
        except Exception as e:
            logger.error(f"❌ Binance connection failed: {e}")
            self.connected = False
            raise
    
    async def _sync_time(self):
        """Sync system time with Binance server time"""
        try:
            server_time = await self.exchange.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = server_time - local_time
            
            if abs(time_diff) > 5000:  # More than 5 seconds difference
                logger.warning(f"⚠️ Time difference with Binance: {time_diff}ms")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not sync time with Binance: {e}")
    
    async def _maintain_websocket_connection(self):
        """Maintain WebSocket connection for real-time data"""
        while self.connected:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    logger.info("🔗 Binance WebSocket connected")
                    
                    # Subscribe to ticker updates for active symbols
                    if self.ws_subscriptions:
                        subscription = {
                            "method": "SUBSCRIBE",
                            "params": list(self.ws_subscriptions),
                            "id": 1
                        }
                        await websocket.send(json.dumps(subscription))
                    
                    # Listen for messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._process_websocket_message(data)
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.warning(f"🔄 Binance WebSocket disconnected: {e}")
                await asyncio.sleep(5)  # Reconnect after 5 seconds
    
    async def _process_websocket_message(self, data: Dict):
        """Process incoming WebSocket messages"""
        if 'stream' in data:
            stream_name = data['stream']
            stream_data = data['data']
            
            if '@ticker' in stream_name:
                # Process ticker update
                symbol = stream_data.get('s')
                logger.debug(f"📊 Ticker update for {symbol}: {stream_data.get('c')}")
    
    async def subscribe_to_ticker(self, symbol: str):
        """Subscribe to real-time ticker updates"""
        stream = f"{symbol.lower()}@ticker"
        self.ws_subscriptions.add(stream)
        logger.info(f"📡 Subscribed to {symbol} ticker updates")
    
    async def fetch_order_book_enhanced(self, symbol: str, limit: int = 100) -> OrderBook:
        """Fetch enhanced order book with validation"""
        await self._check_rate_limit('market')
        
        try:
            raw_book = await self.exchange.fetch_order_book(symbol, limit)
            
            # Convert to enhanced format
            bids = [OrderBookLevel(price=bid[0], quantity=bid[1]) for bid in raw_book['bids']]
            asks = [OrderBookLevel(price=ask[0], quantity=ask[1]) for ask in raw_book['asks']]
            
            order_book = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=raw_book['timestamp'] / 1000 if raw_book['timestamp'] else time.time()
            )
            
            # Validate order book
            if not self._validate_order_book(order_book):
                raise ValueError(f"Invalid order book for {symbol}")
            
            return order_book
            
        except Exception as e:
            logger.error(f"❌ Enhanced order book fetch error for {symbol}: {e}")
            raise
    
    def _validate_order_book(self, order_book: OrderBook) -> bool:
        """Validate order book data integrity"""
        # Check if bids are descending
        for i in range(len(order_book.bids) - 1):
            if order_book.bids[i].price <= order_book.bids[i + 1].price:
                logger.warning(f"⚠️ Invalid bid ordering in {order_book.symbol}")
                return False
        
        # Check if asks are ascending  
        for i in range(len(order_book.asks) - 1):
            if order_book.asks[i].price >= order_book.asks[i + 1].price:
                logger.warning(f"⚠️ Invalid ask ordering in {order_book.symbol}")
                return False
        
        # Check spread
        if order_book.bids and order_book.asks:
            spread = order_book.asks[0].price - order_book.bids[0].price
            if spread <= 0:
                logger.warning(f"⚠️ Invalid spread in {order_book.symbol}: {spread}")
                return False
        
        return True
    
    async def create_advanced_order(self, symbol: str, side: str, amount: float, 
                                  order_type: str = 'market', price: float = None,
                                  stop_price: float = None, time_in_force: str = 'GTC',
                                  reduce_only: bool = False) -> Dict:
        """Create advanced order with multiple order types"""
        await self._check_rate_limit('order')
        
        params = {
            'timeInForce': time_in_force,
            'newOrderRespType': 'FULL'  # Get full order response
        }
        
        if reduce_only:
            params['reduceOnly'] = True
        
        if stop_price:
            params['stopPrice'] = stop_price
        
        try:
            if order_type.lower() == 'market':
                order = await self.exchange.create_market_order(symbol, side, amount, None, None, params)
            elif order_type.lower() == 'limit':
                if not price:
                    raise ValueError("Price required for limit orders")
                order = await self.exchange.create_limit_order(symbol, side, amount, price, params)
            elif order_type.lower() == 'stop_loss':
                if not stop_price:
                    raise ValueError("Stop price required for stop loss orders")
                order = await self.exchange.create_order(symbol, 'STOP_LOSS', side, amount, None, params)
            elif order_type.lower() == 'stop_loss_limit':
                if not price or not stop_price:
                    raise ValueError("Both price and stop price required for stop loss limit orders")
                params['price'] = price
                order = await self.exchange.create_order(symbol, 'STOP_LOSS_LIMIT', side, amount, None, params)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            logger.info(f"📝 Created {order_type} order: {symbol} {side} {amount} @ {price or 'market'}")
            return order
            
        except Exception as e:
            logger.error(f"❌ Advanced order creation failed: {e}")
            raise
    
    async def get_account_summary(self) -> Dict:
        """Get comprehensive account summary"""
        if not self.api_key or not self.secret:
            logger.warning("⚠️ API credentials required for account summary")
            return {}
        
        try:
            # Fetch account information
            balance = await self.exchange.fetch_balance()
            
            # Fetch open orders
            open_orders = await self.exchange.fetch_open_orders()
            
            # Fetch recent trades
            symbols = ['BTCUSDT', 'ETHUSDT']  # Main trading pairs
            recent_trades = {}
            
            for symbol in symbols:
                try:
                    trades = await self.exchange.fetch_my_trades(symbol, limit=10)
                    recent_trades[symbol] = trades
                except:
                    continue  # Skip if no permission or trades
            
            return {
                'balance': balance,
                'open_orders': open_orders,
                'recent_trades': recent_trades,
                'total_open_orders': len(open_orders),
                'account_type': 'spot',
                'trading_enabled': balance.get('info', {}).get('canTrade', False),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Account summary error: {e}")
            return {'error': str(e)}
    
    async def disconnect(self):
        """Enhanced disconnect with WebSocket cleanup"""
        self.connected = False
        
        # Close WebSocket connections
        for ws in self.ws_connections.values():
            try:
                await ws.close()
            except:
                pass
        
        # Close exchange connection
        if self.exchange:
            await self.exchange.close()
            
        logger.info("🔌 Disconnected from Binance")

class MultiExchangeConnector:
    """Connector that aggregates multiple exchanges for best execution"""
    
    def __init__(self):
        self.exchanges = {}
        self.primary_exchange = None
        
    async def add_exchange(self, name: str, connector: ExchangeConnector):
        """Add exchange connector"""
        self.exchanges[name] = connector
        await connector.connect()
        
        if not self.primary_exchange:
            self.primary_exchange = name
            
        logger.info(f"📈 Added exchange: {name}")
    
    async def get_best_price(self, symbol: str, side: str) -> Tuple[str, float]:
        """Get best price across all connected exchanges"""
        best_exchange = None
        best_price = None
        
        for name, exchange in self.exchanges.items():
            try:
                ticker = await exchange.fetch_ticker(symbol)
                
                if side.lower() == 'buy':
                    price = ticker['ask']  # We pay the ask when buying
                    if best_price is None or price < best_price:
                        best_price = price
                        best_exchange = name
                else:
                    price = ticker['bid']  # We receive the bid when selling
                    if best_price is None or price > best_price:
                        best_price = price
                        best_exchange = name
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not get price from {name}: {e}")
                continue
        
        return best_exchange, best_price
    
    async def execute_on_best_exchange(self, symbol: str, side: str, amount: float, order_type: str = 'market'):
        """Execute order on the exchange with the best price"""
        best_exchange_name, best_price = await self.get_best_price(symbol, side)
        
        if best_exchange_name:
            exchange = self.exchanges[best_exchange_name]
            logger.info(f"🎯 Executing on {best_exchange_name} with best price: {best_price}")
            
            if order_type.lower() == 'market':
                return await exchange.create_market_order(symbol, side, amount)
            else:
                return await exchange.create_limit_order(symbol, side, amount, best_price)
        else:
            raise Exception("No exchanges available for execution")
