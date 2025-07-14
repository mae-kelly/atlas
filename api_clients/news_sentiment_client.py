import asyncio
import aiohttp
import requests
from newsapi import NewsApiClient
import praw
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from newspaper import Article
import feedparser
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Optional
import time
from datetime import datetime, timedelta
import os
from loguru import logger
from dataclasses import dataclass
import json

@dataclass
class NewsArticle:
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    sentiment_score: float
    sentiment_label: str
    keywords: List[str]
    author: Optional[str] = None
    
@dataclass
class SentimentAnalysis:
    text: str
    compound_score: float
    positive: float
    negative: float
    neutral: float
    confidence: float
    keywords_detected: List[str]
    source: str

class NewsAndSentimentClient:
    """Comprehensive news and sentiment analysis client"""
    
    def __init__(self):
        # API keys
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT')
        
        # Initialize clients
        self.news_client = NewsApiClient(api_key=self.news_api_key) if self.news_api_key else None
        
        if self.reddit_client_id and self.reddit_client_secret:
            self.reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
        else:
            self.reddit = None
        
        # Sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Keywords for different sectors
        self.sector_keywords = {
            'crypto': ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'nft', 'altcoin'],
            'tech': ['technology', 'ai', 'artificial intelligence', 'software', 'tech stocks'],
            'finance': ['federal reserve', 'interest rates', 'banking', 'financial', 'economy'],
            'energy': ['oil', 'gas', 'renewable energy', 'solar', 'wind power'],
            'healthcare': ['healthcare', 'pharmaceutical', 'biotech', 'medical']
        }
        
        # RSS feeds for additional news sources
        self.rss_feeds = {
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline'
        }
    
    async def get_news_articles(self, query: str, language: str = 'en', 
                               sort_by: str = 'publishedAt', page_size: int = 100) -> List[NewsArticle]:
        """Fetch news articles from News API"""
        if not self.news_client:
            logger.warning("⚠️ News API key not configured")
            return []
        
        try:
            # Fetch from News API
            articles = self.news_client.get_everything(
                q=query,
                language=language,
                sort_by=sort_by,
                page_size=page_size
            )
            
            news_articles = []
            for article_data in articles['articles']:
                try:
                    # Extract and analyze content
                    content = article_data.get('content', '') or article_data.get('description', '')
                    title = article_data.get('title', '')
                    
                    # Perform sentiment analysis
                    full_text = f"{title} {content}"
                    sentiment = self._analyze_text_sentiment(full_text)
                    
                    # Extract keywords
                    keywords = self._extract_keywords(full_text)
                    
                    news_article = NewsArticle(
                        title=title,
                        content=content,
                        url=article_data.get('url', ''),
                        source=article_data.get('source', {}).get('name', 'Unknown'),
                        published_at=datetime.fromisoformat(article_data.get('publishedAt', '').replace('Z', '+00:00')),
                        sentiment_score=sentiment['compound'],
                        sentiment_label=sentiment['label'],
                        keywords=keywords,
                        author=article_data.get('author')
                    )
                    
                    news_articles.append(news_article)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing article: {e}")
                    continue
            
            logger.info(f"📰 Fetched {len(news_articles)} articles for query: {query}")
            return news_articles
            
        except Exception as e:
            logger.error(f"❌ News API error: {e}")
            return []
    
    async def get_reddit_sentiment(self, subreddit_name: str, limit: int = 100) -> List[SentimentAnalysis]:
        """Fetch and analyze sentiment from Reddit"""
        if not self.reddit:
            logger.warning("⚠️ Reddit API credentials not configured")
            return []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            sentiments = []
            
            # Get hot posts
            for submission in subreddit.hot(limit=limit):
                try:
                    # Analyze title
                    title_sentiment = self._analyze_text_sentiment(submission.title)
                    
                    # Get keywords
                    keywords = self._extract_keywords(submission.title)
                    
                    sentiment_analysis = SentimentAnalysis(
                        text=submission.title,
                        compound_score=title_sentiment['compound'],
                        positive=title_sentiment['positive'],
                        negative=title_sentiment['negative'],
                        neutral=title_sentiment['neutral'],
                        confidence=title_sentiment['confidence'],
                        keywords_detected=keywords,
                        source=f'reddit_{subreddit_name}'
                    )
                    
                    sentiments.append(sentiment_analysis)
                    
                    # Analyze top comments
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:5]:  # Top 5 comments
                        if hasattr(comment, 'body') and len(comment.body) > 20:
                            comment_sentiment = self._analyze_text_sentiment(comment.body)
                            comment_keywords = self._extract_keywords(comment.body)
                            
                            comment_analysis = SentimentAnalysis(
                                text=comment.body[:200] + "..." if len(comment.body) > 200 else comment.body,
                                compound_score=comment_sentiment['compound'],
                                positive=comment_sentiment['positive'],
                                negative=comment_sentiment['negative'],
                                neutral=comment_sentiment['neutral'],
                                confidence=comment_sentiment['confidence'],
                                keywords_detected=comment_keywords,
                                source=f'reddit_{subreddit_name}_comment'
                            )
                            
                            sentiments.append(comment_analysis)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing Reddit post: {e}")
                    continue
            
            logger.info(f"📱 Analyzed {len(sentiments)} posts/comments from r/{subreddit_name}")
            return sentiments
            
        except Exception as e:
            logger.error(f"❌ Reddit API error: {e}")
            return []
    
    async def get_rss_news(self, feeds: List[str] = None) -> List[NewsArticle]:
        """Fetch news from RSS feeds"""
        if feeds is None:
            feeds = list(self.rss_feeds.keys())
        
        all_articles = []
        
        for feed_name in feeds:
            if feed_name not in self.rss_feeds:
                continue
                
            try:
                feed_url = self.rss_feeds[feed_name]
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limit to 20 per feed
                    try:
                        # Extract content
                        content = entry.get('summary', '') or entry.get('description', '')
                        title = entry.get('title', '')
                        
                        # Parse date
                        published_at = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_at = datetime(*entry.published_parsed[:6])
                        
                        # Sentiment analysis
                        full_text = f"{title} {content}"
                        sentiment = self._analyze_text_sentiment(full_text)
                        
                        # Keywords
                        keywords = self._extract_keywords(full_text)
                        
                        article = NewsArticle(
                            title=title,
                            content=content,
                            url=entry.get('link', ''),
                            source=feed_name,
                            published_at=published_at,
                            sentiment_score=sentiment['compound'],
                            sentiment_label=sentiment['label'],
                            keywords=keywords,
                            author=entry.get('author')
                        )
                        
                        all_articles.append(article)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing RSS entry: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"❌ RSS feed error for {feed_name}: {e}")
                continue
        
        logger.info(f"📡 Fetched {len(all_articles)} articles from RSS feeds")
        return all_articles
    
    async def scrape_article_content(self, url: str) -> Optional[str]:
        """Scrape full article content from URL"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            return article.text
            
        except Exception as e:
            logger.warning(f"⚠️ Could not scrape article {url}: {e}")
            return None
    
    def _analyze_text_sentiment(self, text: str) -> Dict:
        """Comprehensive sentiment analysis"""
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # Combine scores (weighted average)
        compound_score = (vader_scores['compound'] * 0.6 + textblob_polarity * 0.4)
        
        # Determine label
        if compound_score >= 0.05:
            label = 'positive'
        elif compound_score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Calculate confidence based on agreement
        agreement = abs(vader_scores['compound'] - textblob_polarity)
        confidence = max(0.1, 1.0 - agreement)
        
        return {
            'compound': compound_score,
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'label': label,
            'confidence': confidence,
            'textblob_polarity': textblob_polarity,
            'vader_compound': vader_scores['compound']
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        
        # Check for sector keywords
        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
        
        # Extract other significant words (simplified)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text_lower)
        significant_words = [word for word in words if len(word) > 4 and word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'doesnt', 'let', 'put', 'say', 'she', 'too', 'use']]
        
        # Add most frequent significant words
        from collections import Counter
        word_counts = Counter(significant_words)
        found_keywords.extend([word for word, count in word_counts.most_common(5)])
        
        return list(set(found_keywords))[:10]  # Return unique keywords, max 10
    
    async def get_sector_sentiment_summary(self, sector: str = 'crypto') -> Dict:
        """Get comprehensive sentiment summary for a sector"""
        if sector not in self.sector_keywords:
            logger.error(f"❌ Unknown sector: {sector}")
            return {}
        
        keywords = self.sector_keywords[sector]
        query = ' OR '.join(keywords[:3])  # Use top 3 keywords for news search
        
        # Gather data from multiple sources
        news_articles = await self.get_news_articles(query, page_size=50)
        rss_articles = await self.get_rss_news()
        
        # Filter RSS articles for sector relevance
        relevant_rss = [
            article for article in rss_articles
            if any(keyword in article.title.lower() or keyword in article.content.lower() 
                   for keyword in keywords)
        ]
        
        # Get Reddit sentiment if available
        reddit_sentiments = []
        if sector == 'crypto':
            reddit_sentiments = await self.get_reddit_sentiment('cryptocurrency', limit=50)
        elif sector == 'tech':
            reddit_sentiments = await self.get_reddit_sentiment('technology', limit=50)
        
        # Combine all data
        all_articles = news_articles + relevant_rss
        
        if not all_articles and not reddit_sentiments:
            return {'error': f'No data found for sector: {sector}'}
        
        # Calculate aggregate sentiment
        article_sentiments = [article.sentiment_score for article in all_articles]
        reddit_scores = [sentiment.compound_score for sentiment in reddit_sentiments]
        
        all_sentiments = article_sentiments + reddit_scores
        
        if not all_sentiments:
            return {'error': 'No sentiment data available'}
        
        # Calculate metrics
        avg_sentiment = sum(all_sentiments) / len(all_sentiments)
        sentiment_volatility = (sum((s - avg_sentiment) ** 2 for s in all_sentiments) / len(all_sentiments)) ** 0.5
        
        positive_count = sum(1 for s in all_sentiments if s > 0.05)
        negative_count = sum(1 for s in all_sentiments if s < -0.05)
        neutral_count = len(all_sentiments) - positive_count - negative_count
        
        # Recent trend (last 24 hours)
        recent_articles = [
            article for article in all_articles
            if (datetime.now() - article.published_at).total_seconds() < 86400
        ]
        
        recent_sentiment = 0.0
        if recent_articles:
            recent_sentiment = sum(article.sentiment_score for article in recent_articles) / len(recent_articles)
        
        return {
            'sector': sector,
            'timestamp': time.time(),
            'overall_sentiment': avg_sentiment,
            'sentiment_volatility': sentiment_volatility,
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count,
                'positive_ratio': positive_count / len(all_sentiments),
                'negative_ratio': negative_count / len(all_sentiments)
            },
            'recent_sentiment_24h': recent_sentiment,
            'data_sources': {
                'news_articles': len(news_articles),
                'rss_articles': len(relevant_rss),
                'reddit_posts': len(reddit_sentiments)
            },
            'top_keywords': list(set([kw for article in all_articles for kw in article.keywords]))[:10],
            'recent_headlines': [article.title for article in recent_articles[:5]]
        }
