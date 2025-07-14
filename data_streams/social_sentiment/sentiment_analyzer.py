from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class SentimentAnalyzer:
    """
    Standalone sentiment analysis utility
    """
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def clean_text(self, text: str) -> str:
        """Clean text for better sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions and hashtags symbols (keep the words)
        text = re.sub(r'[@#]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def analyze_batch(self, texts: list) -> dict:
        """
        Analyze sentiment for a batch of texts
        """
        results = []
        
        for text in texts:
            cleaned = self.clean_text(text)
            
            # TextBlob
            blob = TextBlob(cleaned)
            tb_polarity = blob.sentiment.polarity
            
            # VADER
            vader_scores = self.vader.polarity_scores(cleaned)
            
            results.append({
                'text': text,
                'cleaned': cleaned,
                'textblob_polarity': tb_polarity,
                'vader_compound': vader_scores['compound'],
                'consensus': self._consensus(tb_polarity, vader_scores['compound'])
            })
        
        return {
            'individual_results': results,
            'batch_summary': self._summarize_batch(results)
        }
    
    def _consensus(self, tb_polarity: float, vader_compound: float) -> str:
        avg = (tb_polarity + vader_compound) / 2
        if avg > 0.1:
            return 'bullish'
        elif avg < -0.1:
            return 'bearish'
        return 'neutral'
    
    def _summarize_batch(self, results: list) -> dict:
        total = len(results)
        bullish = sum(1 for r in results if r['consensus'] == 'bullish')
        bearish = sum(1 for r in results if r['consensus'] == 'bearish')
        neutral = total - bullish - bearish
        
        return {
            'total_analyzed': total,
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'bullish_ratio': bullish / total if total > 0 else 0,
            'bearish_ratio': bearish / total if total > 0 else 0,
            'sentiment_score': (bullish - bearish) / total if total > 0 else 0
        }
