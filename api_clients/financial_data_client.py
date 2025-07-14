import asyncio
import aiohttp
import yfinance as yf
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta
import os
from loguru import logger
from dataclasses import dataclass

@dataclass
class MarketData:
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str

class FinancialDataClient:
    """Comprehensive financial data client with multiple API sources"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.iex_key = os.getenv('IEX_CLOUD_API_KEY')
        
        # Initialize clients
        self.av_ts = TimeSeries(key=self.alpha_vantage_key) if self.alpha_vantage_key else None
        self.av_fd = FundamentalData(key=self.alpha_vantage_key) if self.alpha_vantage_key else None
        
        # Rate limiting
        self.last_request_times = {}
        self.rate_limits = {
            'alpha_vantage': 5.0,  # 5 requests per minute
            'twelve_data': 0.125,  # 8 requests per second
            'polygon': 0.02,       # 50 requests per second
            'finnhub': 1.0,        # 60 requests per minute
            'iex': 0.1            # 10 requests per second
        }
    
    async def _rate_limit(self, source: str):
        """Apply rate limiting for different data sources"""
        now = time.time()
        if source in self.last_request_times:
            time_since_last = now - self.last_request_times[source]
            min_interval = self.rate_limits.get(source, 1.0)
            
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_times[source] = time.time()
    
    async def get_stock_data_yfinance(self, symbol: str, period: str = "1d", interval: str = "1m") -> List[MarketData]:
        """Fetch stock data using yfinance (free, no API key required)"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            market_data = []
            for timestamp, row in data.iterrows():
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=timestamp.timestamp(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    source='yfinance'
                ))
            
            logger.info(f"📈 Fetched {len(market_data)} data points for {symbol} from yfinance")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ yfinance error for {symbol}: {e}")
            return []
    
    async def get_stock_data_alpha_vantage(self, symbol: str, interval: str = "1min") -> List[MarketData]:
        """Fetch stock data using Alpha Vantage"""
        if not self.av_ts:
            logger.warning("⚠️ Alpha Vantage API key not configured")
            return []
        
        await self._rate_limit('alpha_vantage')
        
        try:
            data, meta_data = self.av_ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
            
            market_data = []
            for timestamp_str, values in data.items():
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').timestamp()
                
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(values['1. open']),
                    high=float(values['2. high']),
                    low=float(values['3. low']),
                    close=float(values['4. close']),
                    volume=float(values['5. volume']),
                    source='alpha_vantage'
                ))
            
            logger.info(f"📈 Fetched {len(market_data)} data points for {symbol} from Alpha Vantage")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Alpha Vantage error for {symbol}: {e}")
            return []
    
    async def get_crypto_data_twelve_data(self, symbol: str, interval: str = "1min") -> List[MarketData]:
        """Fetch crypto data using Twelve Data API"""
        if not self.twelve_data_key:
            logger.warning("⚠️ Twelve Data API key not configured")
            return []
        
        await self._rate_limit('twelve_data')
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.twelvedata.com/time_series"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'apikey': self.twelve_data_key,
                    'outputsize': 5000
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'values' in data:
                            market_data = []
                            for item in data['values']:
                                timestamp = datetime.strptime(item['datetime'], '%Y-%m-%d %H:%M:%S').timestamp()
                                
                                market_data.append(MarketData(
                                    symbol=symbol,
                                    timestamp=timestamp,
                                    open=float(item['open']),
                                    high=float(item['high']),
                                    low=float(item['low']),
                                    close=float(item['close']),
                                    volume=float(item['volume']) if 'volume' in item else 0.0,
                                    source='twelve_data'
                                ))
                            
                            logger.info(f"📈 Fetched {len(market_data)} data points for {symbol} from Twelve Data")
                            return market_data
                        else:
                            logger.error(f"❌ No data in Twelve Data response for {symbol}")
                            return []
                    else:
                        logger.error(f"❌ Twelve Data API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Twelve Data error for {symbol}: {e}")
            return []
    
    async def get_polygon_data(self, symbol: str, timespan: str = "minute", multiplier: int = 1) -> List[MarketData]:
        """Fetch data using Polygon.io API"""
        if not self.polygon_key:
            logger.warning("⚠️ Polygon API key not configured")
            return []
        
        await self._rate_limit('polygon')
        
        try:
            # Get date range (last 2 days for minute data)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                params = {
                    'adjusted': 'true',
                    'sort': 'asc',
                    'limit': 50000,
                    'apikey': self.polygon_key
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'results' in data:
                            market_data = []
                            for item in data['results']:
                                market_data.append(MarketData(
                                    symbol=symbol,
                                    timestamp=item['t'] / 1000,  # Convert from milliseconds
                                    open=float(item['o']),
                                    high=float(item['h']),
                                    low=float(item['l']),
                                    close=float(item['c']),
                                    volume=float(item['v']),
                                    source='polygon'
                                ))
                            
                            logger.info(f"📈 Fetched {len(market_data)} data points for {symbol} from Polygon")
                            return market_data
                        else:
                            logger.error(f"❌ No results in Polygon response for {symbol}")
                            return []
                    else:
                        logger.error(f"❌ Polygon API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Polygon error for {symbol}: {e}")
            return []
    
    async def get_aggregated_data(self, symbol: str, sources: List[str] = None) -> List[MarketData]:
        """Aggregate data from multiple sources with fallback"""
        if sources is None:
            sources = ['yfinance', 'twelve_data', 'polygon', 'alpha_vantage']
        
        all_data = []
        
        for source in sources:
            try:
                if source == 'yfinance':
                    data = await self.get_stock_data_yfinance(symbol)
                elif source == 'alpha_vantage':
                    data = await self.get_stock_data_alpha_vantage(symbol)
                elif source == 'twelve_data':
                    data = await self.get_crypto_data_twelve_data(symbol)
                elif source == 'polygon':
                    data = await self.get_polygon_data(symbol)
                else:
                    continue
                
                if data:
                    all_data.extend(data)
                    logger.info(f"✅ Successfully fetched data from {source}")
                    break  # Use first successful source
                else:
                    logger.warning(f"⚠️ No data from {source}, trying next source...")
                    
            except Exception as e:
                logger.error(f"❌ Error with {source}: {e}")
                continue
        
        if not all_data:
            logger.error(f"❌ Failed to fetch data for {symbol} from any source")
        
        return all_data
    
    async def get_fundamental_data(self, symbol: str) -> Dict:
        """Get fundamental data for a symbol"""
        if not self.av_fd:
            logger.warning("⚠️ Alpha Vantage API key not configured for fundamentals")
            return {}
        
        await self._rate_limit('alpha_vantage')
        
        try:
            # Get company overview
            overview, _ = self.av_fd.get_company_overview(symbol)
            
            # Get key ratios
            ratios = {}
            try:
                income_statement, _ = self.av_fd.get_income_statement_annual(symbol)
                balance_sheet, _ = self.av_fd.get_balance_sheet_annual(symbol)
                
                if income_statement and balance_sheet:
                    latest_income = list(income_statement.values())[0]
                    latest_balance = list(balance_sheet.values())[0]
                    
                    # Calculate key ratios
                    revenue = float(latest_income.get('totalRevenue', 0))
                    net_income = float(latest_income.get('netIncome', 0))
                    total_assets = float(latest_balance.get('totalAssets', 0))
                    
                    ratios = {
                        'revenue': revenue,
                        'net_income': net_income,
                        'profit_margin': (net_income / revenue * 100) if revenue > 0 else 0,
                        'roa': (net_income / total_assets * 100) if total_assets > 0 else 0
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not calculate ratios for {symbol}: {e}")
            
            return {
                'symbol': symbol,
                'overview': overview,
                'ratios': ratios,
                'source': 'alpha_vantage_fundamentals',
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Fundamental data error for {symbol}: {e}")
            return {}

class EconomicDataClient:
    """Client for economic data from FRED and other sources"""
    
    def __init__(self):
        self.fred_key = os.getenv('FRED_API_KEY')
        
    async def get_fred_data(self, series_id: str, limit: int = 100) -> Dict:
        """Fetch data from FRED (Federal Reserve Economic Data)"""
        if not self.fred_key:
            logger.warning("⚠️ FRED API key not configured")
            return {}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_key,
                    'file_type': 'json',
                    'limit': limit,
                    'sort_order': 'desc'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        observations = []
                        for obs in data.get('observations', []):
                            if obs['value'] != '.':  # FRED uses '.' for missing data
                                observations.append({
                                    'date': obs['date'],
                                    'value': float(obs['value']),
                                    'timestamp': datetime.strptime(obs['date'], '%Y-%m-%d').timestamp()
                                })
                        
                        logger.info(f"📊 Fetched {len(observations)} observations for {series_id}")
                        return {
                            'series_id': series_id,
                            'observations': observations,
                            'source': 'fred',
                            'timestamp': time.time()
                        }
                    else:
                        logger.error(f"❌ FRED API error: {response.status}")
                        return {}
                        
        except Exception as e:
            logger.error(f"❌ FRED data error for {series_id}: {e}")
            return {}
    
    async def get_key_economic_indicators(self) -> Dict:
        """Fetch key economic indicators"""
        indicators = {
            'gdp': 'GDP',
            'unemployment': 'UNRATE',
            'inflation': 'CPIAUCSL',
            'interest_rate': 'FEDFUNDS',
            'consumer_sentiment': 'UMCSENT',
            'sp500': 'SP500'
        }
        
        results = {}
        for name, series_id in indicators.items():
            data = await self.get_fred_data(series_id, limit=12)  # Last 12 observations
            if data:
                results[name] = data
                
        return results
