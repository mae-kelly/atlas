import sys
import asyncio
import aiohttp
import torch
import numpy as np
from transformers import pipeline
import time
import json
import re

class M1Sentiment:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        if torch.backends.mps.is_available():
            device = "mps"
        
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
            return_all_scores=True
        )
        self.session = None
        
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def fetch_reddit_data(self, symbol):
        try:
            session = await self.get_session()
            url = f"https://www.reddit.com/search.json?q={symbol}+crypto&sort=new&limit=20&t=hour"
            async with session.get(url, headers={'User-Agent': 'Bot/1.0'}) as response:
                if response.status == 200:
                    data = await response.json()
                    texts = []
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        title = post_data.get('title', '')
                        text = post_data.get('selftext', '')
                        combined = f"{title} {text}".strip()
                        if len(combined) > 10:
                            texts.append(combined[:300])
                    return texts
        except:
            pass
        return []

    async def fetch_crypto_news(self, symbol):
        try:
            session = await self.get_session()
            url = f"https://api.coingecko.com/api/v3/search/trending"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return [f"{symbol} trending crypto"] if symbol.lower() in str(data).lower() else []
        except:
            pass
        return []

    def preprocess_text(self, texts):
        if not texts:
            return []
        
        processed = []
        for text in texts:
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
            if len(text) > 5:
                processed.append(text[:200])
        
        return processed

    def calculate_sentiment(self, texts):
        if not texts:
            return 0.5
        
        try:
            results = self.sentiment_pipeline(texts, truncation=True, max_length=200)
            
            scores = []
            for result in results:
                if isinstance(result, list):
                    result = result[0]
                
                label = result.get('label', '').upper()
                score = result.get('score', 0.0)
                
                if 'POSITIVE' in label or 'LABEL_2' in label:
                    sentiment_score = score
                elif 'NEGATIVE' in label or 'LABEL_0' in label:
                    sentiment_score = -score
                else:
                    sentiment_score = 0.0
                
                scores.append(sentiment_score)
            
            if torch.backends.mps.is_available():
                scores_tensor = torch.tensor(scores, device='mps', dtype=torch.float32)
                weights = torch.exp(torch.arange(len(scores_tensor), device='mps', dtype=torch.float32) * 0.1)
                weights = weights / torch.sum(weights)
                weighted_sentiment = torch.sum(scores_tensor * weights)
                final_score = float(weighted_sentiment.cpu())
            else:
                scores_array = np.array(scores, dtype=np.float32)
                weights = np.exp(np.arange(len(scores_array), dtype=np.float32) * 0.1)
                weights = weights / np.sum(weights)
                final_score = float(np.sum(scores_array * weights))
            
            return max(-1.0, min(1.0, final_score))
            
        except Exception:
            return 0.5

    async def analyze_token(self, symbol):
        try:
            reddit_texts = await self.fetch_reddit_data(symbol)
            news_texts = await self.fetch_crypto_news(symbol)
            
            all_texts = reddit_texts + news_texts
            processed_texts = self.preprocess_text(all_texts)
            
            if not processed_texts:
                return 0.5
            
            sentiment_score = self.calculate_sentiment(processed_texts)
            
            volume_boost = min(len(processed_texts) / 50.0, 0.2)
            final_score = sentiment_score + volume_boost
            
            normalized_score = (final_score + 1.0) / 2.0
            normalized_score = max(0.0, min(1.0, normalized_score))
            
            return normalized_score
            
        except Exception:
            return 0.5

async def main():
    if len(sys.argv) < 2:
        print("0.5")
        return
    
    symbol = sys.argv[1]
    analyzer = M1Sentiment()
    
    try:
        score = await analyzer.analyze_token(symbol)
        print(f"{score:.4f}")
    except Exception:
        print("0.5")
    finally:
        if analyzer.session:
            await analyzer.session.close()

if __name__ == "__main__":
    asyncio.run(main())