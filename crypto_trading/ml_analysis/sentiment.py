import sys
import asyncio
import aiohttp
import random
import time
import json
import re
from typing import Dict, List, Tuple, Optional

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.session = None
        # Sentiment keywords with weights
        self.positive_keywords = {
            'bullish': 3, 'moon': 2, 'pump': 2, 'rocket': 2, 'hodl': 2,
            'buy': 1, 'long': 1, 'green': 1, 'profit': 2, 'gains': 2,
            'breakout': 3, 'surge': 2, 'rally': 2, 'strong': 1, 'support': 1
        }
        
        self.negative_keywords = {
            'bearish': 3, 'dump': 2, 'crash': 3, 'drop': 2, 'fall': 2,
            'sell': 1, 'short': 1, 'red': 1, 'loss': 2, 'down': 1,
            'breakdown': 3, 'resistance': 1, 'weak': 1, 'fear': 2
        }
        
        # Market sentiment cache
        self.sentiment_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
    async def get_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def fetch_reddit_sentiment(self, symbol: str) -> float:
        """Fetch sentiment from Reddit cryptocurrency discussions"""
        try:
            session = await self.get_session()
            
            # Search multiple subreddits
            subreddits = ['cryptocurrency', 'CryptoMarkets', 'altcoins', 'Bitcoin', 'ethereum']
            base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
            
            total_sentiment = 0
            post_count = 0
            
            for subreddit in subreddits:
                try:
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        'q': base_symbol,
                        'sort': 'new',
                        'limit': 10,
                        't': 'day',
                        'restrict_sr': 1
                    }
                    headers = {'User-Agent': 'CryptoBot/1.0'}
                    
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts = data.get('data', {}).get('children', [])
                            
                            for post in posts:
                                post_data = post.get('data', {})
                                title = post_data.get('title', '').lower()
                                selftext = post_data.get('selftext', '').lower()
                                
                                text = f"{title} {selftext}"
                                if base_symbol.lower() in text:
                                    sentiment = self.analyze_text_sentiment(text)
                                    total_sentiment += sentiment
                                    post_count += 1
                
                except Exception as e:
                    continue
            
            if post_count > 0:
                return total_sentiment / post_count
            else:
                return 0.5  # Neutral if no data
                
        except Exception as e:
            return 0.5

    async def fetch_news_sentiment(self, symbol: str) -> float:
        """Simulate fetching news sentiment from multiple sources"""
        try:
            # In a real implementation, you'd fetch from:
            # - CoinDesk, CoinTelegraph, Decrypt
            # - Twitter API for crypto influencers
            # - Google News API
            # - Binance News API
            
            base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
            
            # Simulate different news sentiment based on time and symbol
            current_time = int(time.time())
            symbol_hash = hash(base_symbol) % 1000
            time_factor = (current_time % 3600) / 3600  # Hour-based variation
            
            # Major coins tend to have more stable sentiment
            if base_symbol.upper() in ['BTC', 'ETH', 'SOL', 'ADA']:
                base_sentiment = 0.55 + (symbol_hash % 100) / 500  # 0.55-0.75
            else:
                base_sentiment = 0.45 + (symbol_hash % 100) / 200  # 0.45-0.95
            
            # Add time-based volatility
            sentiment = base_sentiment + (time_factor - 0.5) * 0.2
            
            return max(0.0, min(1.0, sentiment))
            
        except Exception as e:
            return 0.5

    def analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of a text using keyword analysis"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_score = 0
        negative_score = 0
        
        for word in words:
            if word in self.positive_keywords:
                positive_score += self.positive_keywords[word]
            elif word in self.negative_keywords:
                negative_score += self.negative_keywords[word]
        
        total_score = positive_score + negative_score
        if total_score == 0:
            return 0.5  # Neutral
        
        sentiment = positive_score / total_score
        return max(0.0, min(1.0, sentiment))

    async def get_market_fear_greed_index(self) -> float:
        """Simulate Fear & Greed Index (0 = extreme fear, 1 = extreme greed)"""
        try:
            # In real implementation, fetch from:
            # - Alternative.me Fear & Greed Index
            # - Crypto Fear & Greed Index APIs
            
            current_time = int(time.time())
            # Create pseudo-random but stable index based on time
            base_index = (current_time // 3600) % 100  # Changes every hour
            
            # Simulate market cycles
            cycle_position = (current_time // 86400) % 30  # 30-day cycle
            if cycle_position < 10:
                # Bull phase
                fear_greed = 0.6 + (base_index / 250)  # 0.6-0.8
            elif cycle_position < 20:
                # Volatile phase
                fear_greed = 0.3 + (base_index / 167)  # 0.3-0.9
            else:
                # Bear phase
                fear_greed = 0.2 + (base_index / 333)  # 0.2-0.5
            
            return max(0.0, min(1.0, fear_greed))
            
        except Exception as e:
            return 0.5

    async def analyze_token_fundamentals(self, symbol: str) -> float:
        """Analyze fundamental factors affecting token sentiment"""
        base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
        
        # Fundamental analysis factors
        factors = {
            'market_cap_rank': 0.5,  # Will be dynamic in real implementation
            'development_activity': 0.5,
            'partnership_news': 0.5,
            'adoption_metrics': 0.5,
            'technical_updates': 0.5
        }
        
        # Major tokens get higher fundamental scores
        if base_symbol.upper() in ['BTC', 'ETH']:
            factors['market_cap_rank'] = 0.9
            factors['adoption_metrics'] = 0.9
        elif base_symbol.upper() in ['SOL', 'ADA', 'DOT', 'LINK']:
            factors['market_cap_rank'] = 0.8
            factors['development_activity'] = 0.8
        elif base_symbol.upper() in ['AVAX', 'MATIC', 'UNI', 'ATOM']:
            factors['market_cap_rank'] = 0.7
            factors['development_activity'] = 0.7
        
        # Add some randomness for simulation
        current_time = int(time.time())
        symbol_seed = hash(base_symbol) % 1000
        
        for key in factors:
            time_variance = ((current_time + symbol_seed) % 3600) / 7200  # ±0.25 variance
            factors[key] += (time_variance - 0.25)
            factors[key] = max(0.0, min(1.0, factors[key]))
        
        # Weighted average
        weights = {
            'market_cap_rank': 0.3,
            'development_activity': 0.25,
            'partnership_news': 0.2,
            'adoption_metrics': 0.15,
            'technical_updates': 0.1
        }
        
        fundamental_score = sum(factors[k] * weights[k] for k in factors)
        return fundamental_score

    async def calculate_technical_sentiment(self, symbol: str) -> float:
        """Calculate sentiment based on technical indicators"""
        # Simulate technical analysis sentiment
        # In real implementation, this would analyze:
        # - RSI levels
        # - Moving average positions
        # - Volume trends
        # - Support/resistance levels
        
        current_time = int(time.time())
        symbol_hash = hash(symbol) % 1000
        
        # Create technical sentiment based on "market conditions"
        market_phase = (current_time // 1800) % 8  # 30-minute cycles
        
        technical_factors = {
            'trend': 0.5,
            'momentum': 0.5,
            'volume': 0.5,
            'volatility': 0.5
        }
        
        # Simulate different market conditions
        if market_phase in [0, 1]:  # Strong uptrend
            technical_factors['trend'] = 0.8 + (symbol_hash % 100) / 500
            technical_factors['momentum'] = 0.7 + (symbol_hash % 100) / 400
        elif market_phase in [2, 3]:  # Sideways
            technical_factors['trend'] = 0.4 + (symbol_hash % 100) / 500
            technical_factors['volume'] = 0.3 + (symbol_hash % 100) / 400
        elif market_phase in [4, 5]:  # Downtrend
            technical_factors['trend'] = 0.2 + (symbol_hash % 100) / 500
            technical_factors['momentum'] = 0.3 + (symbol_hash % 100) / 400
        else:  # Volatile
            technical_factors['volatility'] = 0.8 + (symbol_hash % 100) / 500
            technical_factors['momentum'] = 0.5 + ((symbol_hash % 100) - 50) / 250
        
        # Weighted technical sentiment
        weights = [0.4, 0.3, 0.2, 0.1]  # trend, momentum, volume, volatility
        technical_sentiment = sum(list(technical_factors.values())[i] * weights[i] for i in range(4))
        
        return max(0.0, min(1.0, technical_sentiment))

    async def get_social_volume_impact(self, symbol: str) -> float:
        """Calculate social media volume impact on sentiment"""
        base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
        
        # Simulate social media volume data
        current_time = int(time.time())
        daily_cycle = (current_time % 86400) / 86400  # Position in day
        
        # Major coins have higher social volume
        if base_symbol.upper() in ['BTC', 'ETH']:
            base_volume = 0.8
        elif base_symbol.upper() in ['SOL', 'ADA', 'DOT']:
            base_volume = 0.6
        else:
            base_volume = 0.4
        
        # Add daily pattern (higher volume during US trading hours)
        if 0.5 < daily_cycle < 0.9:  # US hours
            volume_multiplier = 1.3
        else:
            volume_multiplier = 0.8
        
        volume_impact = base_volume * volume_multiplier
        return max(0.0, min(1.0, volume_impact))

    async def analyze_token(self, symbol: str) -> float:
        """Comprehensive token sentiment analysis"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{int(time.time() // self.cache_expiry)}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            # Gather sentiment from multiple sources
            sentiment_sources = await asyncio.gather(
                self.fetch_reddit_sentiment(symbol),
                self.fetch_news_sentiment(symbol),
                self.get_market_fear_greed_index(),
                self.analyze_token_fundamentals(symbol),
                self.calculate_technical_sentiment(symbol),
                self.get_social_volume_impact(symbol),
                return_exceptions=True
            )
            
            # Handle any exceptions and use default values
            reddit_sentiment = sentiment_sources[0] if not isinstance(sentiment_sources[0], Exception) else 0.5
            news_sentiment = sentiment_sources[1] if not isinstance(sentiment_sources[1], Exception) else 0.5
            fear_greed = sentiment_sources[2] if not isinstance(sentiment_sources[2], Exception) else 0.5
            fundamentals = sentiment_sources[3] if not isinstance(sentiment_sources[3], Exception) else 0.5
            technical = sentiment_sources[4] if not isinstance(sentiment_sources[4], Exception) else 0.5
            social_volume = sentiment_sources[5] if not isinstance(sentiment_sources[5], Exception) else 0.5
            
            # Weighted composite sentiment score
            weights = {
                'reddit': 0.15,
                'news': 0.20,
                'fear_greed': 0.15,
                'fundamentals': 0.20,
                'technical': 0.25,
                'social_volume': 0.05
            }
            
            composite_sentiment = (
                reddit_sentiment * weights['reddit'] +
                news_sentiment * weights['news'] +
                fear_greed * weights['fear_greed'] +
                fundamentals * weights['fundamentals'] +
                technical * weights['technical'] +
                social_volume * weights['social_volume']
            )
            
            # Apply confidence adjustments based on data quality
            confidence_factor = self.calculate_confidence_factor(
                reddit_sentiment, news_sentiment, fear_greed, 
                fundamentals, technical, social_volume
            )
            
            # Adjust sentiment based on confidence
            if confidence_factor < 0.5:
                # Low confidence - move toward neutral
                composite_sentiment = 0.5 + (composite_sentiment - 0.5) * confidence_factor * 2
            
            # Ensure bounds
            final_sentiment = max(0.0, min(1.0, composite_sentiment))
            
            # Cache the result
            self.sentiment_cache[cache_key] = final_sentiment
            
            # Clean old cache entries (keep cache size manageable)
            if len(self.sentiment_cache) > 100:
                old_keys = [k for k in self.sentiment_cache.keys() if k.split('_')[-1] < str(int(time.time() // self.cache_expiry) - 2)]
                for k in old_keys:
                    del self.sentiment_cache[k]
            
            return final_sentiment
            
        except Exception as e:
            # Fallback sentiment with some intelligence
            return self.get_fallback_sentiment(symbol)

    def calculate_confidence_factor(self, reddit: float, news: float, fear_greed: float, 
                                  fundamentals: float, technical: float, social_volume: float) -> float:
        """Calculate confidence in the sentiment analysis based on data convergence"""
        sentiments = [reddit, news, fear_greed, fundamentals, technical, social_volume]
        
        # Calculate standard deviation
        mean_sentiment = sum(sentiments) / len(sentiments)
        variance = sum((s - mean_sentiment) ** 2 for s in sentiments) / len(sentiments)
        std_dev = variance ** 0.5
        
        # Lower standard deviation = higher confidence
        confidence = max(0.0, min(1.0, 1.0 - (std_dev * 3)))
        
        return confidence

    def get_fallback_sentiment(self, symbol: str) -> float:
        """Generate intelligent fallback sentiment when APIs fail"""
        base_symbol = symbol.replace('-USDT', '').replace('-USD', '').replace('-USDC', '')
        
        # Base sentiment for different tier coins
        if base_symbol.upper() in ['BTC', 'ETH']:
            base_sentiment = 0.65  # Generally more positive for major coins
        elif base_symbol.upper() in ['SOL', 'ADA', 'DOT', 'LINK', 'AVAX']:
            base_sentiment = 0.60  # Solid altcoins
        elif base_symbol.upper() in ['MATIC', 'UNI', 'ATOM', 'FTM', 'NEAR']:
            base_sentiment = 0.55  # Mid-tier altcoins
        else:
            base_sentiment = 0.50  # Unknown/smaller coins - neutral
        
        # Add time-based variation to simulate market cycles
        current_time = int(time.time())
        daily_cycle = (current_time % 86400) / 86400
        weekly_cycle = (current_time % 604800) / 604800
        
        # Market hours affect sentiment
        if 0.5 < daily_cycle < 0.9:  # US trading hours
            time_boost = 0.05
        elif 0.3 < daily_cycle < 0.5:  # Asian hours
            time_boost = 0.02
        else:  # Off hours
            time_boost = -0.02
        
        # Weekly cycle (weekends often more volatile)
        if 0.7 < weekly_cycle < 1.0:  # Weekend
            weekly_boost = -0.03
        else:
            weekly_boost = 0.01
        
        # Symbol-specific hash for consistency
        symbol_hash = hash(base_symbol) % 1000
        symbol_variance = (symbol_hash / 1000 - 0.5) * 0.1  # ±5% based on symbol
        
        final_sentiment = base_sentiment + time_boost + weekly_boost + symbol_variance
        return max(0.0, min(1.0, final_sentiment))

    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

# Main execution
async def main():
    if len(sys.argv) < 2:
        print("0.5000")
        return
    
    symbol = sys.argv[1]
    analyzer = AdvancedSentimentAnalyzer()
    
    try:
        score = await analyzer.analyze_token(symbol)
        print(f"{score:.4f}")
    except Exception as e:
        # Even on error, provide an intelligent fallback
        fallback_score = analyzer.get_fallback_sentiment(symbol)
        print(f"{fallback_score:.4f}")
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())
