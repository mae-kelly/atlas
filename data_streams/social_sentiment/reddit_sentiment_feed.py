import asyncio
import aiohttp
import time
import re
from typing import List, Callable, Dict
from loguru import logger
from collections import deque
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
class RedditSentimentFeed:
    """
    Real-time sentiment analysis from Reddit cryptocurrency discussions
    """
    def __init__(self, on_sentiment_update: Callable):
        self.on_sentiment_update = on_sentiment_update
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.reddit_base = "https://www.reddit.com"
        self.subreddits = [
            'cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets', 
            'altcoin', 'defi', 'web3', 'solana', 'cardano'
        ]
        self.sentiment_history = deque(maxlen=1000)
        self.last_request_time = 0
        self.request_delay = 2.0
        self.crypto_keywords = {
            'bitcoin': ['bitcoin', 'btc', 'satoshi'],
            'ethereum': ['ethereum', 'eth', 'ether', 'vitalik'],
            'cardano': ['cardano', 'ada', 'charles'],
            'solana': ['solana', 'sol'],
            'polkadot': ['polkadot', 'dot'],
            'chainlink': ['chainlink', 'link'],
            'polygon': ['polygon', 'matic'],
            'avalanche': ['avalanche', 'avax'],
            'crypto_general': ['crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft', 'web3']
        }
    async def start_sentiment_stream(self):
        """
        Start streaming sentiment from Reddit
        """
        logger.info("🐦 Starting Reddit sentiment stream...")
        while True:
            try:
                for subreddit in self.subreddits:
                    await self._fetch_subreddit_posts(subreddit)
                    await asyncio.sleep(10)
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"❌ Sentiment stream error: {e}")
                await asyncio.sleep(30)
    async def _fetch_subreddit_posts(self, subreddit: str):
        """
        Fetch recent posts from a subreddit
        """
        await self._rate_limit()
        try:
            url = f"{self.reddit_base}/r/{subreddit}/hot.json"
            headers = {
                'User-Agent': 'CryptoSentiment/1.0 (Educational)'
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('data', {}).get('children', [])
                        for post_data in posts[:10]:
                            post = post_data.get('data', {})
                            await self._analyze_post(post, subreddit)
                    else:
                        logger.warning(f"⚠️ Reddit API returned {response.status} for r/{subreddit}")
        except Exception as e:
            logger.error(f"❌ Error fetching r/{subreddit}: {e}")
    async def _analyze_post(self, post: Dict, subreddit: str):
        """
        Analyze sentiment of a Reddit post
        """
        try:
            title = post.get('title', '')
            selftext = post.get('selftext', '')
            created_utc = post.get('created_utc', time.time())
            score = post.get('score', 0)
            num_comments = post.get('num_comments', 0)
            full_text = f"{title} {selftext}".strip()
            if len(full_text) < 10:
                return
            if time.time() - created_utc > 86400:
                return
            cleaned_text = self._clean_text(full_text)
            sentiment_data = self._analyze_sentiment(cleaned_text)
            keywords = self._extract_crypto_keywords(cleaned_text)
            if not keywords:
                return
            sentiment_event = {
                'text': full_text[:200],  # First 200 chars
                'sentiment_score': sentiment_data['compound'],
                'sentiment_label': sentiment_data['label'],
                'keywords': keywords,
                'confidence': abs(sentiment_data['compound']),
                'source': f'reddit_r_{subreddit}',
                'timestamp': time.time(),
                'metadata': {
                    'post_score': score,
                    'num_comments': num_comments,
                    'subreddit': subreddit,
                    'created_utc': created_utc,
                    'textblob_polarity': sentiment_data.get('textblob_polarity', 0),
                    'vader_scores': sentiment_data.get('vader_details', {})
                }
            }
            self.sentiment_history.append(sentiment_event)
            await self.on_sentiment_update(sentiment_event)
        except Exception as e:
            logger.error(f"❌ Post analysis error: {e}")
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'/u/\w+|/r/\w+', '', text)
        text = ' '.join(text.split())
        return text
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using multiple methods"""
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        vader_scores = self.vader_analyzer.polarity_scores(text)
        consensus_score = (textblob_polarity + vader_scores['compound']) / 2
        if consensus_score > 0.1:
            label = 'bullish'
        elif consensus_score < -0.1:
            label = 'bearish'
        else:
            label = 'neutral'
        return {
            'compound': vader_scores['compound'],
            'label': label,
            'textblob_polarity': textblob_polarity,
            'vader_details': vader_scores
        }
    def _extract_crypto_keywords(self, text: str) -> List[str]:
        """Extract cryptocurrency keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        for crypto, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(crypto)
                    break
        return list(set(found_keywords))
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    def get_sentiment_summary(self, window_minutes: int = 60) -> Dict:
        """Get sentiment summary for the last N minutes"""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_sentiments = [
            s for s in self.sentiment_history 
            if s['timestamp'] >= cutoff_time
        ]
        if not recent_sentiments:
            return {}
        scores = [s['sentiment_score'] for s in recent_sentiments]
        bullish = sum(1 for s in recent_sentiments if s['sentiment_label'] == 'bullish')
        bearish = sum(1 for s in recent_sentiments if s['sentiment_label'] == 'bearish')
        neutral = len(recent_sentiments) - bullish - bearish
        keyword_counts = {}
        for s in recent_sentiments:
            for keyword in s['keywords']:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        return {
            'total_posts': len(recent_sentiments),
            'avg_sentiment': sum(scores) / len(scores),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'sentiment_distribution': {
                'bullish': bullish / len(recent_sentiments),
                'bearish': bearish / len(recent_sentiments),
                'neutral': neutral / len(recent_sentiments)
            },
            'top_keywords': dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'window_minutes': window_minutes
        }