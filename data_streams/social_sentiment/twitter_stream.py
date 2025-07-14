import asyncio
import tweepy
import aiohttp
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
from typing import Callable, List, Optional, Dict, Set
import os
from dotenv import load_dotenv
import time
import json
import re
from collections import deque
import numpy as np
load_dotenv()
class TwitterSentimentStream:
    """
    Enhanced Twitter sentiment stream with Twitter API v2 and multiple analysis engines
    """
    def __init__(self, 
                 keywords: List[str], 
                 on_sentiment: Callable[[dict], None],
                 bearer_token: Optional[str] = None):
        self.keywords = keywords
        self.on_sentiment = on_sentiment
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
        self.consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
        self.consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        if not self.bearer_token:
            raise ValueError("Twitter Bearer Token required. Set TWITTER_BEARER_TOKEN env var.")
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True
        )
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.processed_tweets = deque(maxlen=10000)
        self.seen_tweet_ids = set()
        self.sentiment_buffer = deque(maxlen=1000)
        self.last_request_time = 0
        self.min_request_interval = 1.0
        self.min_followers = 100
        self.exclude_retweets = True
        self.language_filter = ['en']
    async def stream_tweets_v2(self):
        """
        Stream tweets using Twitter API v2 with enhanced filtering
        """
        logger.info(f"🐦 Starting Twitter sentiment stream for: {', '.join(self.keywords)}")
        while True:
            try:
                query = self._build_search_query()
                tweets = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=query,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations', 'lang'],
                    user_fields=['public_metrics', 'verified'],
                    expansions=['author_id'],
                    max_results=100,
                ).flatten(limit=500)
                tweet_count = 0
                for tweet in tweets:
                    try:
                        if tweet.id not in self.seen_tweet_ids:
                            await self._process_tweet_v2(tweet)
                            self.seen_tweet_ids.add(tweet.id)
                            tweet_count += 1
                            if len(self.seen_tweet_ids) > 50000:
                                old_ids = list(self.seen_tweet_ids)[:12500]
                                self.seen_tweet_ids -= set(old_ids)
                    except Exception as e:
                        logger.error(f"❌ Error processing tweet {tweet.id}: {e}")
                logger.info(f"📊 Processed {tweet_count} new tweets")
                await self._calculate_rolling_sentiment()
                await asyncio.sleep(60)
            except tweepy.TooManyRequests:
                logger.warning("⏰ Rate limit reached, waiting 15 minutes...")
                await asyncio.sleep(900)
            except Exception as e:
                logger.warning(f"🔄 Twitter stream error, retrying: {e}")
                await asyncio.sleep(120)
    def _build_search_query(self) -> str:
        """Build optimized search query"""
        keyword_query = ' OR '.join(self.keywords)
        filters = []
        if self.exclude_retweets:
            filters.append('-is:retweet')
        if self.language_filter:
            filters.extend([f'lang:{lang}' for lang in self.language_filter])
        filters.extend(['-is:reply', '-is:quote'])
        full_query = f"({keyword_query}) {' '.join(filters)}"
        return full_query
    async def _process_tweet_v2(self, tweet):
        """Process tweet with enhanced sentiment analysis"""
        try:
            user_info = getattr(tweet, 'includes', {}).get('users', [{}])[0] if hasattr(tweet, 'includes') else {}
            user_followers = user_info.get('public_metrics', {}).get('followers_count', 0)
            if user_followers < self.min_followers and user_followers > 0:
                return
            sentiment_data = await self._analyze_sentiment_enhanced(tweet.text)
            context_data = self._extract_context(tweet)
            influence_score = self._calculate_influence_score(tweet, user_info)
            tweet_data = {
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at.isoformat() if tweet.created_at else None,
                'author_id': tweet.author_id,
                'metrics': tweet.public_metrics,
                'user_metrics': user_info.get('public_metrics', {}),
                'user_verified': user_info.get('verified', False),
                'sentiment': sentiment_data,
                'context': context_data,
                'influence_score': influence_score,
                'keywords_matched': [kw for kw in self.keywords if kw.lower() in tweet.text.lower()],
                'timestamp': time.time(),
                'source': 'twitter_v2'
            }
            self.processed_tweets.append(tweet_data)
            await asyncio.create_task(self.on_sentiment(tweet_data))
        except Exception as e:
            logger.error(f"❌ Error processing tweet: {e}")
    async def _analyze_sentiment_enhanced(self, text: str) -> Dict:
        """Enhanced sentiment analysis with multiple models"""
        cleaned_text = self._clean_text(text)
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
        crypto_sentiment = self._analyze_crypto_sentiment(cleaned_text)
        emoji_sentiment = self._analyze_emoji_sentiment(text)
        combined_score = (
            textblob_polarity * 0.3 +
            vader_scores['compound'] * 0.4 +
            crypto_sentiment * 0.2 +
            emoji_sentiment * 0.1
        )
        confidence = self._calculate_sentiment_confidence(
            textblob_polarity, vader_scores['compound'], crypto_sentiment
        )
        return {
            'textblob': {
                'polarity': textblob_polarity,
                'subjectivity': textblob_subjectivity,
                'label': self._get_sentiment_label(textblob_polarity)
            },
            'vader': {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'label': self._get_sentiment_label(vader_scores['compound'])
            },
            'crypto_sentiment': crypto_sentiment,
            'emoji_sentiment': emoji_sentiment,
            'combined_score': combined_score,
            'confidence': confidence,
            'consensus': self._get_consensus_sentiment(combined_score),
            'cleaned_text': cleaned_text
        }
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = ' '.join(text.split())
        text = re.sub(r'[^\w\s\!\?\.\,\:\;\-\(\)\$\€\£\💰\🚀\📈\📉\😀-\😿]', ' ', text)
        return text.strip()
    def _analyze_crypto_sentiment(self, text: str) -> float:
        """Analyze crypto-specific sentiment keywords"""
        bullish_keywords = [
            'moon', 'mooning', 'bullish', 'bull run', 'hodl', 'diamond hands',
            'to the moon', 'pump', 'rally', 'breakout', 'all time high',
            'ath', 'green', 'gains', 'profit', 'buy the dip'
        ]
        bearish_keywords = [
            'dump', 'crash', 'bear', 'bearish', 'red', 'loss', 'rekt',
            'paper hands', 'sell', 'panic', 'fud', 'fear', 'correction',
            'dip', 'decline', 'drop'
        ]
        text_lower = text.lower()
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in text_lower)
        if bullish_count == 0 and bearish_count == 0:
            return 0.0
        total_count = bullish_count + bearish_count
        sentiment_score = (bullish_count - bearish_count) / total_count
        return sentiment_score
    def _analyze_emoji_sentiment(self, text: str) -> float:
        """Analyze emoji sentiment"""
        positive_emojis = ['🚀', '📈', '💰', '🌙', '💎', '🔥', '💪', '🎉', '✅', '🟢']
        negative_emojis = ['📉', '💸', '😭', '😱', '🔴', '❌', '💀', '🩸', '😰', '📊']
        positive_count = sum(text.count(emoji) for emoji in positive_emojis)
        negative_count = sum(text.count(emoji) for emoji in negative_emojis)
        if positive_count == 0 and negative_count == 0:
            return 0.0
        total_count = positive_count + negative_count
        return (positive_count - negative_count) / total_count
    def _calculate_sentiment_confidence(self, tb_score: float, vader_score: float, crypto_score: float) -> float:
        """Calculate confidence based on sentiment model agreement"""
        scores = [tb_score, vader_score, crypto_score]
        non_zero_scores = [s for s in scores if s != 0]
        if len(non_zero_scores) < 2:
            return 0.3
        std_dev = np.std(non_zero_scores)
        confidence = max(0.1, 1.0 - (std_dev * 2))
        return min(1.0, confidence)
    def _get_sentiment_label(self, score: float) -> str:
        """Convert numerical score to label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    def _get_consensus_sentiment(self, combined_score: float) -> str:
        """Get consensus sentiment for trading"""
        if combined_score > 0.1:
            return 'bullish'
        elif combined_score < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    def _extract_context(self, tweet) -> Dict:
        """Extract contextual information from tweet"""
        context = {
            'has_media': bool(getattr(tweet, 'attachments', {})),
            'reply_count': tweet.public_metrics.get('reply_count', 0),
            'retweet_count': tweet.public_metrics.get('retweet_count', 0),
            'like_count': tweet.public_metrics.get('like_count', 0),
            'quote_count': tweet.public_metrics.get('quote_count', 0),
        }
        if hasattr(tweet, 'context_annotations'):
            context['entities'] = [
                annotation.get('entity', {}).get('name', '')
                for annotation in tweet.context_annotations
            ]
        return context
    def _calculate_influence_score(self, tweet, user_info: Dict) -> float:
        """Calculate tweet influence score"""
        followers = user_info.get('public_metrics', {}).get('followers_count', 0)
        likes = tweet.public_metrics.get('like_count', 0)
        retweets = tweet.public_metrics.get('retweet_count', 0)
        replies = tweet.public_metrics.get('reply_count', 0)
        verified_boost = 1.5 if user_info.get('verified', False) else 1.0
        engagement = likes + (retweets * 2) + (replies * 1.5)
        follower_influence = np.log10(max(1, followers))
        influence = (engagement + follower_influence) * verified_boost
        normalized_influence = min(1.0, influence / 10000)
        return normalized_influence
    async def _calculate_rolling_sentiment(self):
        """Calculate and emit rolling sentiment metrics"""
        if len(self.processed_tweets) < 10:
            return
        current_time = time.time()
        recent_tweets = [
            tweet for tweet in self.processed_tweets
            if current_time - tweet['timestamp'] < 3600
        ]
        if not recent_tweets:
            return
        sentiments = [tweet['sentiment']['combined_score'] for tweet in recent_tweets]
        influences = [tweet['influence_score'] for tweet in recent_tweets]
        if influences:
            weighted_sentiment = np.average(sentiments, weights=influences)
        else:
            weighted_sentiment = np.mean(sentiments)
        bullish_count = sum(1 for s in sentiments if s > 0.1)
        bearish_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - bullish_count - bearish_count
        total_engagement = sum(
            tweet['metrics'].get('like_count', 0) +
            tweet['metrics'].get('retweet_count', 0) +
            tweet['metrics'].get('reply_count', 0)
            for tweet in recent_tweets
        )
        sentiment_report = {
            'timestamp': current_time,
            'time_window_hours': 1,
            'total_tweets': len(recent_tweets),
            'sentiment_score': weighted_sentiment,
            'sentiment_distribution': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count,
                'bullish_ratio': bullish_count / len(sentiments),
                'bearish_ratio': bearish_count / len(sentiments)
            },
            'engagement_metrics': {
                'total_engagement': total_engagement,
                'avg_engagement_per_tweet': total_engagement / len(recent_tweets),
                'high_influence_tweets': sum(1 for i in influences if i > 0.5)
            },
            'keywords': self.keywords,
            'source': 'twitter_rolling_sentiment'
        }
        self.sentiment_buffer.append(sentiment_report)
        try:
            await asyncio.create_task(self.on_sentiment(sentiment_report))
        except Exception as e:
            logger.error(f"❌ Error emitting rolling sentiment: {e}")
        logger.info(f"📈 Rolling Sentiment: {weighted_sentiment:.3f} "
                   f"({bullish_count}🟢 {bearish_count}🔴 {neutral_count}⚪)")
class TwitterSentimentAggregator:
    """Enhanced sentiment aggregation with advanced analytics"""
    def __init__(self, buffer_size: int = 1000):
        self.sentiment_buffer = deque(maxlen=buffer_size)
        self.rolling_metrics = deque(maxlen=100)
    async def process_sentiment(self, tweet_data: dict):
        """Process and aggregate sentiment data with enhanced analytics"""
        self.sentiment_buffer.append(tweet_data)
        if tweet_data.get('source') == 'twitter_rolling_sentiment':
            self._log_rolling_sentiment(tweet_data)
            return
        sentiment = tweet_data['sentiment']['consensus']
        influence = tweet_data.get('influence_score', 0)
        keywords = ', '.join(tweet_data['keywords_matched'])
        logger.info(f"📊 [{sentiment.upper()}] {keywords} (influence: {influence:.2f}): "
                   f"{tweet_data['text'][:100]}...")
        if len(self.sentiment_buffer) % 50 == 0:
            await self._update_running_metrics()
    def _log_rolling_sentiment(self, sentiment_report: dict):
        """Log rolling sentiment report"""
        score = sentiment_report['sentiment_score']
        distribution = sentiment_report['sentiment_distribution']
        direction = "📈 BULLISH" if score > 0.1 else "📉 BEARISH" if score < -0.1 else "➡️ NEUTRAL"
        logger.info(f"{direction} Rolling Sentiment: {score:.3f}")
        logger.info(f"   Distribution: {distribution['bullish']}🟢 "
                   f"{distribution['bearish']}🔴 {distribution['neutral']}⚪")
        logger.info(f"   Total Tweets: {sentiment_report['total_tweets']}")
        logger.info(f"   Engagement: {sentiment_report['engagement_metrics']['total_engagement']:,}")
    async def _update_running_metrics(self):
        """Update running sentiment metrics"""
        if len(self.sentiment_buffer) < 20:
            return
        recent_tweets = list(self.sentiment_buffer)[-100:]
        sentiments = [t['sentiment']['combined_score'] for t in recent_tweets]
        influences = [t.get('influence_score', 0) for t in recent_tweets]
        metrics = {
            'timestamp': time.time(),
            'sample_size': len(recent_tweets),
            'avg_sentiment': np.mean(sentiments),
            'sentiment_volatility': np.std(sentiments),
            'weighted_sentiment': np.average(sentiments, weights=influences) if influences else np.mean(sentiments),
            'high_influence_ratio': sum(1 for i in influences if i > 0.3) / len(influences)
        }
        self.rolling_metrics.append(metrics)
        logger.info(f"🔄 Updated Metrics: Avg Sentiment: {metrics['avg_sentiment']:.3f}, "
                   f"Volatility: {metrics['sentiment_volatility']:.3f}")