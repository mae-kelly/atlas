import asyncio
import aiohttp
import ccxt.async_support as ccxt
from pycoingecko import CoinGeckoAPI
import cryptocompare
import requests
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta
import os
from loguru import logger
from dataclasses import dataclass
import json
@dataclass
class CryptoPrice:
    symbol: str
    price_usd: float
    price_btc: float
    volume_24h: float
    market_cap: float
    percent_change_24h: float
    timestamp: float
    source: str
@dataclass
class DeFiMetrics:
    protocol: str
    tvl: float
    volume_24h: float
    fees_24h: float
    revenue_24h: float
    token_price: float
    market_cap: float
    timestamp: float
class CryptocurrencyClient:
    """Comprehensive cryptocurrency data client"""
    def __init__(self):
        self.coingecko_key = os.getenv('COINGECKO_API_KEY')
        self.cryptocompare_key = os.getenv('CRYPTOCOMPARE_API_KEY')
        self.binance_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
        self.cg = CoinGeckoAPI()
        if self.cryptocompare_key:
            cryptocompare.cryptocompare._set_api_key_parameter(self.cryptocompare_key)
        self.exchanges = {}
        self.last_request_times = {}
        self.rate_limits = {
            'coingecko': 1.0,      # 50 requests per minute for free tier
            'cryptocompare': 0.1,   # 100 requests per second for free tier
            'binance': 0.1,        # Weight-based limiting
            'defillama': 1.0       # Conservative limit
        }
    async def _rate_limit(self, source: str):
        """Apply rate limiting"""
        now = time.time()
        if source in self.last_request_times:
            time_since_last = now - self.last_request_times[source]
            min_interval = self.rate_limits.get(source, 1.0)
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
        self.last_request_times[source] = time.time()
    async def get_top_cryptocurrencies(self, limit: int = 100) -> List[CryptoPrice]:
        """Get top cryptocurrencies by market cap from CoinGecko"""
        await self._rate_limit('coingecko')
        try:
            data = self.cg.get_coins_markets(
                vs_currency='usd',
                order='market_cap_desc',
                per_page=limit,
                page=1,
                sparkline=False,
                price_change_percentage='24h'
            )
            crypto_prices = []
            for coin in data:
                try:
                    btc_price = self.cg.get_price(ids='bitcoin', vs_currencies='usd')['bitcoin']['usd']
                    price_btc = coin['current_price'] / btc_price if btc_price > 0 else 0
                    crypto_price = CryptoPrice(
                        symbol=coin['symbol'].upper(),
                        price_usd=coin['current_price'] or 0.0,
                        price_btc=price_btc,
                        volume_24h=coin['total_volume'] or 0.0,
                        market_cap=coin['market_cap'] or 0.0,
                        percent_change_24h=coin['price_change_percentage_24h'] or 0.0,
                        timestamp=time.time(),
                        source='coingecko'
                    )
                    crypto_prices.append(crypto_price)
                except Exception as e:
                    logger.warning(f"⚠️ Error processing coin {coin.get('id', 'unknown')}: {e}")
                    continue
            logger.info(f"💰 Fetched {len(crypto_prices)} cryptocurrency prices from CoinGecko")
            return crypto_prices
        except Exception as e:
            logger.error(f"❌ CoinGecko error: {e}")
            return []
    async def get_defi_protocols(self, limit: int = 50) -> List[DeFiMetrics]:
        """Get DeFi protocol data from DeFiLlama"""
        await self._rate_limit('defillama')
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.llama.fi/protocols"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        defi_metrics = []
                        for protocol in data[:limit]:
                            try:
                                volume_24h = 0.0
                                fees_24h = 0.0
                                revenue_24h = 0.0
                                token_price = 0.0
                                market_cap = 0.0
                                if 'gecko_id' in protocol and protocol['gecko_id']:
                                    try:
                                        token_data = self.cg.get_coin_by_id(protocol['gecko_id'])
                                        token_price = token_data['market_data']['current_price']['usd']
                                        market_cap = token_data['market_data']['market_cap']['usd']
                                    except:
                                        pass
                                defi_metric = DeFiMetrics(
                                    protocol=protocol['name'],
                                    tvl=protocol.get('tvl', 0.0),
                                    volume_24h=volume_24h,
                                    fees_24h=fees_24h,
                                    revenue_24h=revenue_24h,
                                    token_price=token_price,
                                    market_cap=market_cap,
                                    timestamp=time.time()
                                )
                                defi_metrics.append(defi_metric)
                            except Exception as e:
                                logger.warning(f"⚠️ Error processing DeFi protocol: {e}")
                                continue
                        logger.info(f"🏦 Fetched {len(defi_metrics)} DeFi protocols from DeFiLlama")
                        return defi_metrics
                    else:
                        logger.error(f"❌ DeFiLlama API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"❌ DeFiLlama error: {e}")
            return []
    async def get_crypto_fear_greed_index(self) -> Dict:
        """Get Crypto Fear & Greed Index"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.alternative.me/fng/"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            latest = data['data'][0]
                            return {
                                'value': int(latest['value']),
                                'value_classification': latest['value_classification'],
                                'timestamp': int(latest['timestamp']),
                                'time_until_update': latest.get('time_until_update'),
                                'source': 'alternative.me'
                            }
                    logger.error(f"❌ Fear & Greed API error: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"❌ Fear & Greed Index error: {e}")
            return {}
    async def get_crypto_dominance(self) -> Dict:
        """Get cryptocurrency market dominance data"""
        await self._rate_limit('coingecko')
        try:
            global_data = self.cg.get_global()
            dominance = {}
            if 'market_cap_percentage' in global_data:
                for symbol, percentage in global_data['market_cap_percentage'].items():
                    dominance[symbol.upper()] = percentage
            total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
            total_volume = global_data.get('total_volume', {}).get('usd', 0)
            return {
                'dominance': dominance,
                'total_market_cap_usd': total_market_cap,
                'total_volume_24h_usd': total_volume,
                'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
                'markets': global_data.get('markets', 0),
                'timestamp': time.time(),
                'source': 'coingecko'
            }
        except Exception as e:
            logger.error(f"❌ Crypto dominance error: {e}")
            return {}
    async def get_historical_prices(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical price data for a cryptocurrency"""
        await self._rate_limit('coingecko')
        try:
            coins_list = self.cg.get_coins_list()
            coin_id = None
            for coin in coins_list:
                if coin['symbol'].upper() == symbol.upper():
                    coin_id = coin['id']
                    break
            if not coin_id:
                logger.error(f"❌ Could not find coin ID for symbol: {symbol}")
                return []
            data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=days
            )
            historical_data = []
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0
                historical_data.append({
                    'timestamp': timestamp / 1000,  # Convert from milliseconds
                    'date': datetime.fromtimestamp(timestamp / 1000).isoformat(),
                    'price': price,
                    'volume': volume,
                    'symbol': symbol.upper()
                })
            logger.info(f"📈 Fetched {len(historical_data)} historical data points for {symbol}")
            return historical_data
        except Exception as e:
            logger.error(f"❌ Historical price error for {symbol}: {e}")
            return []
    async def get_exchange_info(self, exchange_name: str = 'binance') -> Dict:
        """Get exchange information and trading pairs"""
        try:
            if exchange_name.lower() == 'binance':
                exchange = ccxt.binance({
                    'apiKey': self.binance_key,
                    'secret': self.binance_secret,
                    'sandbox': False,
                    'enableRateLimit': True
                })
                await exchange.load_markets()
                tickers = await exchange.fetch_tickers()
                exchange_info = {
                    'name': exchange_name,
                    'total_pairs': len(exchange.markets),
                    'active_pairs': len([symbol for symbol, market in exchange.markets.items() if market['active']]),
                    'base_currencies': list(set([market['base'] for market in exchange.markets.values()])),
                    'quote_currencies': list(set([market['quote'] for market in exchange.markets.values()])),
                    'top_volume_pairs': [],
                    'timestamp': time.time()
                }
                if tickers:
                    sorted_pairs = sorted(
                        [(symbol, ticker) for symbol, ticker in tickers.items() if ticker['quoteVolume']],
                        key=lambda x: x[1]['quoteVolume'],
                        reverse=True
                    )
                    exchange_info['top_volume_pairs'] = [
                        {
                            'symbol': symbol,
                            'volume_24h': ticker['quoteVolume'],
                            'price': ticker['last'],
                            'change_24h': ticker['percentage']
                        }
                        for symbol, ticker in sorted_pairs[:20]
                    ]
                await exchange.close()
                return exchange_info
        except Exception as e:
            logger.error(f"❌ Exchange info error for {exchange_name}: {e}")
            return {}
    async def get_on_chain_metrics(self, symbol: str = 'BTC') -> Dict:
        """Get on-chain metrics for cryptocurrencies"""
        try:
            return {
                'symbol': symbol,
                'active_addresses': 0,
                'transaction_count_24h': 0,
                'hash_rate': 0,
                'network_value_to_transactions': 0,
                'mvrv_ratio': 0,
                'realized_price': 0,
                'hodl_waves': {},
                'exchange_flows': {
                    'inflow_24h': 0,
                    'outflow_24h': 0,
                    'net_flow': 0
                },
                'timestamp': time.time(),
                'source': 'placeholder'
            }
        except Exception as e:
            logger.error(f"❌ On-chain metrics error for {symbol}: {e}")
            return {}
    async def get_comprehensive_crypto_data(self, symbols: List[str] = None) -> Dict:
        """Get comprehensive cryptocurrency market overview"""
        if symbols is None:
            symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'MATIC', 'AVAX']
        try:
            top_cryptos = await self.get_top_cryptocurrencies(limit=100)
            requested_cryptos = [crypto for crypto in top_cryptos if crypto.symbol in symbols]
            dominance_data = await self.get_crypto_dominance()
            fear_greed = await self.get_crypto_fear_greed_index()
            defi_protocols = await self.get_defi_protocols(limit=20)
            total_market_cap = sum(crypto.market_cap for crypto in top_cryptos[:100])
            total_volume_24h = sum(crypto.volume_24h for crypto in top_cryptos[:100])
            positive_performers = len([crypto for crypto in top_cryptos if crypto.percent_change_24h > 0])
            negative_performers = len([crypto for crypto in top_cryptos if crypto.percent_change_24h < 0])
            return {
                'market_overview': {
                    'total_market_cap': dominance_data.get('total_market_cap_usd', total_market_cap),
                    'total_volume_24h': dominance_data.get('total_volume_24h_usd', total_volume_24h),
                    'bitcoin_dominance': dominance_data.get('dominance', {}).get('BTC', 0),
                    'ethereum_dominance': dominance_data.get('dominance', {}).get('ETH', 0),
                    'fear_greed_index': fear_greed.get('value', 50),
                    'fear_greed_classification': fear_greed.get('value_classification', 'Neutral'),
                    'positive_performers': positive_performers,
                    'negative_performers': negative_performers
                },
                'requested_cryptocurrencies': [
                    {
                        'symbol': crypto.symbol,
                        'price_usd': crypto.price_usd,
                        'market_cap': crypto.market_cap,
                        'volume_24h': crypto.volume_24h,
                        'change_24h': crypto.percent_change_24h
                    }
                    for crypto in requested_cryptos
                ],
                'defi_overview': {
                    'total_protocols': len(defi_protocols),
                    'total_tvl': sum(protocol.tvl for protocol in defi_protocols),
                    'top_protocols': [
                        {
                            'name': protocol.protocol,
                            'tvl': protocol.tvl,
                            'token_price': protocol.token_price
                        }
                        for protocol in sorted(defi_protocols, key=lambda x: x.tvl, reverse=True)[:10]
                    ]
                },
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"❌ Comprehensive crypto data error: {e}")
            return {'error': 'Failed to fetch comprehensive crypto data'}