import asyncio
import time
import json
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
@dataclass
class PriceEvent:
    symbol: str
    price: float
    volume: float
    timestamp: float
    source: str = "binance"
@dataclass
class SentimentEvent:
    keywords: List[str]
    sentiment_score: float
    consensus: str
    confidence: float
    text_sample: str
    timestamp: float
    source: str = "twitter"
@dataclass
class FusionSignal:
    symbol: str
    signal_type: str
    strength: float
    direction: str
    price_component: float
    sentiment_component: float
    correlation_score: float
    timestamp: float
    metadata: Dict
class DataFusionEngine:
    """
    FULLY FIXED: Core engine that fuses price and sentiment data
    """
    def __init__(self, 
                 price_buffer_size: int = 1000,
                 sentiment_buffer_size: int = 500,
                 correlation_window: int = 60):
        self.price_buffer = defaultdict(lambda: deque(maxlen=price_buffer_size))
        self.sentiment_buffer = deque(maxlen=sentiment_buffer_size)
        self.correlation_window = correlation_window
        self.signal_handlers = []
        self.correlation_history = deque(maxlen=100)
        self.metrics = {
            'events_processed': 0,
            'signals_generated': 0,
            'avg_correlation': 0.0,
            'last_signal_time': 0
        }
        logger.info("🧬 Data Fusion Engine initialized (FULLY FIXED)")
    def add_signal_handler(self, handler: Callable[[FusionSignal], None]):
        """Add a callback for when signals are generated"""
        self.signal_handlers.append(handler)
    async def ingest_price_event(self, event: PriceEvent):
        """Process incoming price data"""
        self.price_buffer[event.symbol].append(event)
        self.metrics['events_processed'] += 1
        await self._analyze_correlations(event.symbol)
    async def ingest_sentiment_event(self, event: SentimentEvent):
        """Process incoming sentiment data"""
        self.sentiment_buffer.append(event)
        self.metrics['events_processed'] += 1
        relevant_symbols = self._extract_symbols_from_keywords(event.keywords)
        for symbol in relevant_symbols:
            await self._analyze_correlations(symbol)
    def _extract_symbols_from_keywords(self, keywords: List[str]) -> List[str]:
        """Map keywords to trading symbols"""
        keyword_to_symbol = {
            'bitcoin': 'BTCUSDT',
            'btc': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'eth': 'ETHUSDT',
            'crypto': ['BTCUSDT', 'ETHUSDT'],
            'cryptocurrency': ['BTCUSDT', 'ETHUSDT']
        }
        symbols = set()
        for keyword in keywords:
            mapped = keyword_to_symbol.get(keyword.lower())
            if mapped:
                if isinstance(mapped, list):
                    symbols.update(mapped)
                else:
                    symbols.add(mapped)
        return list(symbols) if symbols else ['BTCUSDT']
    async def _analyze_correlations(self, symbol: str):
        """FULLY FIXED: Analyze price-sentiment correlations for a symbol"""
        current_time = time.time()
        price_data = self._get_recent_price_data(symbol, current_time)
        if len(price_data) < 3:
            return
        sentiment_data = self._get_recent_sentiment_data(current_time)
        if len(sentiment_data) < 2:
            return
        correlation_result = self._calculate_correlations_fixed(price_data, sentiment_data)
        if correlation_result:
            signals = await self._generate_signals(symbol, correlation_result, current_time)
            for signal in signals:
                await self._emit_signal(signal)
    def _get_recent_price_data(self, symbol: str, current_time: float) -> List[PriceEvent]:
        """Get price data within the correlation window"""
        cutoff_time = current_time - self.correlation_window
        return [
            event for event in self.price_buffer[symbol]
            if event.timestamp >= cutoff_time
        ]
    def _get_recent_sentiment_data(self, current_time: float) -> List[SentimentEvent]:
        """Get sentiment data within the correlation window"""
        cutoff_time = current_time - self.correlation_window
        return [
            event for event in self.sentiment_buffer
            if event.timestamp >= cutoff_time
        ]
    def _calculate_correlations_fixed(self, price_data: List[PriceEvent], 
                                    sentiment_data: List[SentimentEvent]) -> Optional[Dict]:
        """FULLY FIXED: Calculate price-sentiment correlations"""
        try:
            if len(price_data) < 2 or len(sentiment_data) < 2:
                return None
            prices = [event.price for event in price_data]
            if len(prices) < 2:
                return None
            price_returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    price_returns.append(ret)
            if len(price_returns) == 0:
                return None
            sentiment_scores = []
            for i, price_event in enumerate(price_data[1:], 1):
                closest_sentiment = None
                min_time_diff = float('inf')
                for sentiment_event in sentiment_data:
                    time_diff = abs(sentiment_event.timestamp - price_event.timestamp)
                    if time_diff < min_time_diff and time_diff < 30:
                        min_time_diff = time_diff
                        closest_sentiment = sentiment_event
                if closest_sentiment:
                    sentiment_scores.append(closest_sentiment.sentiment_score)
                else:
                    sentiment_scores.append(0.0)
            min_length = min(len(price_returns), len(sentiment_scores))
            if min_length < 2:
                return None
            price_returns = price_returns[:min_length]
            sentiment_scores = sentiment_scores[:min_length]
            price_array = np.array(price_returns)
            sentiment_array = np.array(sentiment_scores)
            if np.std(price_array) == 0 or np.std(sentiment_array) == 0:
                correlation = 0.0
            else:
                try:
                    corr_matrix = np.corrcoef(price_array, sentiment_array)
                    if corr_matrix.shape == (2, 2):
                        correlation = float(corr_matrix[0, 1])
                        if np.isnan(correlation):
                            correlation = 0.0
                    else:
                        correlation = 0.0
                except:
                    correlation = 0.0
            price_momentum = float(np.mean(price_array)) if len(price_array) > 0 else 0.0
            sentiment_momentum = float(np.mean(sentiment_array)) if len(sentiment_array) > 0 else 0.0
            result = {
                'correlation': correlation,
                'price_momentum': price_momentum,
                'sentiment_momentum': sentiment_momentum,
                'price_volatility': float(np.std(price_array)) if len(price_array) > 0 else 0.0,
                'sentiment_volatility': float(np.std(sentiment_array)) if len(sentiment_array) > 0 else 0.0,
                'data_points': min_length,
                'latest_price': price_data[-1].price,
                'latest_sentiment': sentiment_data[-1].sentiment_score if sentiment_data else 0.0
            }
            self.correlation_history.append({
                'timestamp': time.time(),
                'correlation': correlation,
                'price_momentum': price_momentum,
                'sentiment_momentum': sentiment_momentum
            })
            return result
        except Exception as e:
            logger.error(f"❌ FIXED Correlation calculation error: {e}")
            return None
    async def _generate_signals(self, symbol: str, correlation_data: Dict, 
                              timestamp: float) -> List[FusionSignal]:
        """Generate trading signals based on correlation analysis"""
        signals = []
        correlation = correlation_data['correlation']
        price_momentum = correlation_data['price_momentum']
        sentiment_momentum = correlation_data['sentiment_momentum']
        if abs(correlation) > 0.3:
            alignment_strength = min(abs(correlation), 1.0)
            if (price_momentum > 0 and sentiment_momentum > 0) or \
               (price_momentum < 0 and sentiment_momentum < 0):
                direction = "bullish" if price_momentum > 0 else "bearish"
                signal = FusionSignal(
                    symbol=symbol,
                    signal_type="momentum_sentiment_alignment",
                    strength=alignment_strength,
                    direction=direction,
                    price_component=abs(price_momentum),
                    sentiment_component=abs(sentiment_momentum),
                    correlation_score=correlation,
                    timestamp=timestamp,
                    metadata={
                        'price_momentum': price_momentum,
                        'sentiment_momentum': sentiment_momentum,
                        'data_points': correlation_data['data_points']
                    }
                )
                signals.append(signal)
        if abs(correlation) > 0.2 and \
           ((price_momentum > 0.001 and sentiment_momentum < -0.1) or 
            (price_momentum < -0.001 and sentiment_momentum > 0.1)):
            divergence_strength = abs(price_momentum - sentiment_momentum)
            signal = FusionSignal(
                symbol=symbol,
                signal_type="sentiment_price_divergence",
                strength=min(divergence_strength, 1.0),
                direction="contrarian",
                price_component=abs(price_momentum),
                sentiment_component=abs(sentiment_momentum),
                correlation_score=correlation,
                timestamp=timestamp,
                metadata={
                    'divergence_type': 'price_leading' if abs(price_momentum) > abs(sentiment_momentum) else 'sentiment_leading',
                    'price_momentum': price_momentum,
                    'sentiment_momentum': sentiment_momentum
                }
            )
            signals.append(signal)
        return signals
    async def _emit_signal(self, signal: FusionSignal):
        """Emit signal to all registered handlers"""
        self.metrics['signals_generated'] += 1
        self.metrics['last_signal_time'] = signal.timestamp
        logger.info(f"🚨 SIGNAL: {signal.signal_type} for {signal.symbol} - "
                   f"{signal.direction.upper()} (strength: {signal.strength:.2f})")
        for handler in self.signal_handlers:
            try:
                await handler(signal)
            except Exception as e:
                logger.error(f"❌ Signal handler error: {e}")
    def get_correlation_summary(self) -> Dict:
        """FIXED: Get recent correlation metrics"""
        if not self.correlation_history:
            return {
                'avg_correlation': 0.0,
                'correlation_volatility': 0.0,
                'recent_correlations': [],
                'total_signals': self.metrics['signals_generated'],
                'events_processed': self.metrics['events_processed']
            }
        history_list = list(self.correlation_history)
        recent_correlations = [item['correlation'] for item in history_list[-20:]]
        return {
            'avg_correlation': float(np.mean(recent_correlations)) if recent_correlations else 0.0,
            'correlation_volatility': float(np.std(recent_correlations)) if len(recent_correlations) > 1 else 0.0,
            'recent_correlations': [float(c) for c in recent_correlations[-5:]],
            'total_signals': self.metrics['signals_generated'],
            'events_processed': self.metrics['events_processed']
        }
class SignalLogger:
    """Helper class to log and analyze signals"""
    def __init__(self):
        self.signals_log = []
    async def log_signal(self, signal: FusionSignal):
        """Log signal for analysis"""
        self.signals_log.append(asdict(signal))
        print(f"\n🎯 NEW SIGNAL DETECTED")
        print(f"Symbol: {signal.symbol}")
        print(f"Type: {signal.signal_type}")
        print(f"Direction: {signal.direction}")
        print(f"Strength: {signal.strength:.3f}")
        print(f"Correlation: {signal.correlation_score:.3f}")
        print(f"Price Component: {signal.price_component:.4f}")
        print(f"Sentiment Component: {signal.sentiment_component:.3f}")
        print(f"Metadata: {signal.metadata}")
        print("=" * 50)
    def get_signal_summary(self) -> Dict:
        """Get summary of logged signals"""
        if not self.signals_log:
            return {"total_signals": 0}
        try:
            df = pd.DataFrame(self.signals_log)
            return {
                "total_signals": len(self.signals_log),
                "signal_types": dict(df['signal_type'].value_counts()),
                "direction_distribution": dict(df['direction'].value_counts()),
                "avg_strength": float(df['strength'].mean()),
                "avg_correlation": float(df['correlation_score'].mean()),
                "recent_signals": self.signals_log[-5:]
            }
        except Exception as e:
            logger.error(f"❌ Signal summary error: {e}")
            return {
                "total_signals": len(self.signals_log),
                "recent_signals": self.signals_log[-5:] if self.signals_log else []
            }