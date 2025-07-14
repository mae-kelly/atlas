import numpy as np,pandas as pd,asyncio
from typing import Dict,List
from .crypto_bert import CryptoBERT,AspectLevelSentiment,MultilingualSentiment,InfluencerWeightedSentiment
from loguru import logger
import time

class AdvancedSentimentAggregator:
    def __init__(self):
        self.crypto_bert=CryptoBERT()
        self.aspect_analyzer=AspectLevelSentiment()
        self.multilingual_analyzer=MultilingualSentiment()
        self.influencer_weighted=InfluencerWeightedSentiment()
        self.sentiment_history=[]
        self.correlation_cache={}
    
    async def process_social_media_batch(self,social_data:List[Dict])->Dict:
        logger.info(f"📱 Processing {len(social_data)} social media posts")
        results={'crypto_bert':[],'aspect_level':[],'multilingual':[],'influencer_weighted':[],'aggregated_metrics':{}}
        texts=[item['text']for item in social_data]
        crypto_bert_results=self.crypto_bert.analyze_crypto_sentiment(texts)
        results['crypto_bert']=crypto_bert_results
        for i,item in enumerate(social_data):
            aspect_results=self.aspect_analyzer.analyze_aspects(item['text'])
            multilingual_results=self.multilingual_analyzer.analyze_multilingual(item['text'])
            results['aspect_level'].append(aspect_results)
            results['multilingual'].append(multilingual_results)
        enhanced_data=[{**social_data[i],'sentiment_score':crypto_bert_results[i]['final_sentiment']['sentiment_score'],'confidence':crypto_bert_results[i]['confidence']}for i in range(len(social_data))]
        results['influencer_weighted']=self.influencer_weighted.weight_by_influence(enhanced_data)
        results['aggregated_metrics']=self._calculate_aggregated_metrics(results)
        return results
    
    def _calculate_aggregated_metrics(self,results:Dict)->Dict:
        crypto_sentiments=[r['final_sentiment']['sentiment_score']for r in results['crypto_bert']]
        weighted_sentiments=[r['weighted_sentiment']for r in results['influencer_weighted']]
        aspect_sentiments={}
        for aspect_result in results['aspect_level']:
            for aspect,data in aspect_result.items():
                if aspect not in aspect_sentiments:aspect_sentiments[aspect]=[]
                sentiment_score=1 if data['sentiment']['label']=='POSITIVE'else-1 if data['sentiment']['label']=='NEGATIVE'else 0
                aspect_sentiments[aspect].append(sentiment_score*data['sentiment']['score'])
        return{'overall_sentiment':np.mean(crypto_sentiments),'weighted_sentiment':np.mean(weighted_sentiments),'sentiment_volatility':np.std(crypto_sentiments),'bullish_ratio':sum(1 for s in crypto_sentiments if s>0.1)/len(crypto_sentiments),'bearish_ratio':sum(1 for s in crypto_sentiments if s<-0.1)/len(crypto_sentiments),'aspect_sentiments':{aspect:np.mean(scores)for aspect,scores in aspect_sentiments.items()if scores},'confidence_weighted_sentiment':np.average(crypto_sentiments,[r['confidence']for r in results['crypto_bert']]),'total_analyzed':len(crypto_sentiments)}
    
    async def correlate_sentiment_with_price(self,sentiment_data:Dict,price_data:pd.DataFrame,symbol:str='BTC')->Dict:
        if symbol not in self.correlation_cache:self.correlation_cache[symbol]=[]
        timestamp=time.time()
        sentiment_score=sentiment_data['aggregated_metrics']['overall_sentiment']
        self.correlation_cache[symbol].append({'timestamp':timestamp,'sentiment':sentiment_score})
        if len(self.correlation_cache[symbol])>100:self.correlation_cache[symbol]=self.correlation_cache[symbol][-100:]
        if len(self.correlation_cache[symbol])<10:return{'correlation':0,'sample_size':len(self.correlation_cache[symbol])}
        sentiment_series=[item['sentiment']for item in self.correlation_cache[symbol]]
        if len(price_data)>=len(sentiment_series):
            recent_returns=price_data['close'].pct_change().tail(len(sentiment_series)).values
            correlation=np.corrcoef(sentiment_series,recent_returns)[0,1]if len(sentiment_series)>1 else 0
            return{'correlation':correlation if not np.isnan(correlation)else 0,'sample_size':len(sentiment_series),'sentiment_mean':np.mean(sentiment_series),'price_volatility':np.std(recent_returns)}
        return{'correlation':0,'sample_size':len(sentiment_series)}
    
    def generate_sentiment_signals(self,aggregated_data:Dict)->List[Dict]:
        signals=[]
        metrics=aggregated_data['aggregated_metrics']
        overall_sentiment=metrics['overall_sentiment']
        confidence=metrics.get('confidence_weighted_sentiment',overall_sentiment)
        volatility=metrics['sentiment_volatility']
        if overall_sentiment>0.3 and confidence>0.5:
            signals.append({'signal_type':'bullish_sentiment','strength':min(overall_sentiment*confidence,1.0),'confidence':confidence,'source':'aggregated_sentiment'})
        elif overall_sentiment<-0.3 and confidence>0.5:
            signals.append({'signal_type':'bearish_sentiment','strength':min(abs(overall_sentiment)*confidence,1.0),'confidence':confidence,'source':'aggregated_sentiment'})
        if volatility>0.5:
            signals.append({'signal_type':'high_sentiment_volatility','strength':min(volatility,1.0),'confidence':0.7,'source':'sentiment_volatility'})
        bullish_ratio,bearish_ratio=metrics['bullish_ratio'],metrics['bearish_ratio']
        if bullish_ratio>0.7:
            signals.append({'signal_type':'sentiment_consensus_bullish','strength':bullish_ratio,'confidence':0.8,'source':'sentiment_distribution'})
        elif bearish_ratio>0.7:
            signals.append({'signal_type':'sentiment_consensus_bearish','strength':bearish_ratio,'confidence':0.8,'source':'sentiment_distribution'})
        return signals
