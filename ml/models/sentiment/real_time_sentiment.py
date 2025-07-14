import asyncio,aiohttp,json,time
from typing import Dict,List
from .sentiment_aggregator import AdvancedSentimentAggregator
from loguru import logger
class RealTimeSentimentMonitor:
    def __init__(self,update_interval:int=30):
        self.aggregator=AdvancedSentimentAggregator()
        self.update_interval,self.running=update_interval,False
        self.sentiment_callbacks,self.signal_callbacks=[],[]
        self.data_sources={'twitter':self._fetch_twitter_data,'reddit':self._fetch_reddit_data,'news':self._fetch_news_data,'telegram':self._fetch_telegram_data}
    async def start_monitoring(self,symbols:List[str]=['bitcoin','ethereum']):
        self.running=True
        logger.info(f"👁️ Starting real-time sentiment monitoring for {symbols}")
        while self.running:
            try:
                all_social_data=[]
                for source_name,fetch_func in self.data_sources.items():
                    try:
                        source_data=await fetch_func(symbols)
                        all_social_data.extend(source_data)
                        logger.debug(f"📊 Fetched {len(source_data)} items from {source_name}")
                    except Exception as e:logger.error(f"❌ Error fetching from {source_name}: {e}")
                if all_social_data:
                    sentiment_results=await self.aggregator.process_social_media_batch(all_social_data)
                    signals=self.aggregator.generate_sentiment_signals(sentiment_results)
                    await self._notify_sentiment_update(sentiment_results)
                    await self._notify_signal_update(signals)
                    logger.info(f"📈 Processed {len(all_social_data)} social posts, generated {len(signals)} signals")
                await asyncio.sleep(self.update_interval)
            except Exception as e:logger.error(f"❌ Sentiment monitoring error: {e}");await asyncio.sleep(5)
    async def _fetch_twitter_data(self,symbols:List[str])->List[Dict]:
        mock_data=[]
        for symbol in symbols:
            for i in range(10):
                mock_data.append({'text':f"{symbol} looking bullish today! 🚀 Great potential for gains",'author':f'crypto_trader_{i}','follower_count':1000+i*100,'timestamp':time.time()-i*60,'source':'twitter'})
        return mock_data
    async def _fetch_reddit_data(self,symbols:List[str])->List[Dict]:
        mock_data=[]
        for symbol in symbols:
            for i in range(5):
                sentiments=['bullish analysis shows strong fundamentals','bearish trend might continue','neutral outlook for next week']
                mock_data.append({'text':f"{symbol} {sentiments[i%3]}",'author':f'reddit_user_{i}','follower_count':500,'timestamp':time.time()-i*120,'source':'reddit'})
        return mock_data
    async def _fetch_news_data(self,symbols:List[str])->List[Dict]:
        return[{'text':f"{symbol} partnership announcement drives positive sentiment",'author':'coindesk','follower_count':50000,'timestamp':time.time(),'source':'news'}for symbol in symbols]
    async def _fetch_telegram_data(self,symbols:List[str])->List[Dict]:
        return[{'text':f"{symbol} moon mission starting! Diamond hands only 💎",'author':'crypto_whale','follower_count':5000,'timestamp':time.time(),'source':'telegram'}for symbol in symbols]
    def add_sentiment_callback(self,callback):
        self.sentiment_callbacks.append(callback)
    def add_signal_callback(self,callback):
        self.signal_callbacks.append(callback)
    async def _notify_sentiment_update(self,sentiment_data:Dict):
        for callback in self.sentiment_callbacks:
            try:await callback(sentiment_data)
            except Exception as e:logger.error(f"❌ Sentiment callback error: {e}")
    async def _notify_signal_update(self,signals:List[Dict]):
        for callback in self.signal_callbacks:
            try:await callback(signals)
            except Exception as e:logger.error(f"❌ Signal callback error: {e}")
    def stop_monitoring(self):
        self.running=False
        logger.info("⏹️ Sentiment monitoring stopped")