import torch,torch.nn as nn,numpy as np,re
from transformers import AutoTokenizer,AutoModel
class EmotionSentimentAnalyzer:
    def __init__(self):
        self.emotion_lexicon={'joy':0.8,'fear':-0.7,'anger':-0.8,'sadness':-0.6,'surprise':0.3,'disgust':-0.7}
        self.crypto_emotions={'moon':0.9,'rekt':-0.9,'hodl':0.5,'fud':-0.8,'fomo':0.6,'diamond_hands':0.7}
    def analyze_emotions(self,text):
        emotions={'joy':0,'fear':0,'anger':0,'sadness':0,'surprise':0,'disgust':0}
        words=text.lower().split()
        for word in words:
            for emotion in emotions:
                if word in self._get_emotion_words(emotion):emotions[emotion]+=1
        total_emotions=sum(emotions.values())
        return{k:v/total_emotions for k,v in emotions.items()}if total_emotions>0 else emotions
    def _get_emotion_words(self,emotion):
        word_map={'joy':['happy','excited','great','amazing','awesome'],'fear':['scared','worried','afraid','panic'],'anger':['angry','mad','furious','hate'],'sadness':['sad','depressed','down','disappointed'],'surprise':['wow','shocked','unbelievable'],'disgust':['disgusting','terrible','awful','horrible']}
        return word_map.get(emotion,[])
class MultilingualCryptoSentiment:
    def __init__(self):
        self.language_models={'en':'bert-base-uncased','zh':'bert-base-chinese','ja':'bert-base-japanese','ko':'bert-base-multilingual-cased'}
        self.crypto_terms_by_lang={'en':['bitcoin','cryptocurrency','blockchain'],'zh':['比特币','加密货币'],'ja':['ビットコイン','暗号通貨'],'ko':['비트코인','암호화폐']}
    def detect_language(self,text):
        char_patterns={'en':r'[a-zA-Z]','zh':r'[\u4e00-\u9fff]','ja':r'[ひらがなカタカナ]','ko':r'[ㄱ-ㅎㅏ-ㅣ가-힣]'}
        scores={lang:len(re.findall(pattern,text))for lang,pattern in char_patterns.items()}
        return max(scores,key=scores.get)if max(scores.values())>0 else'en'
    def analyze_multilingual_sentiment(self,text):
        lang=self.detect_language(text);base_sentiment=self._get_base_sentiment(text,lang)
        crypto_boost=self._calculate_crypto_sentiment_boost(text,lang)
        return{'sentiment':base_sentiment+crypto_boost,'language':lang,'confidence':0.8}
    def _get_base_sentiment(self,text,lang):return 0.1*np.random.randn()
    def _calculate_crypto_sentiment_boost(self,text,lang):
        terms=self.crypto_terms_by_lang.get(lang,[]);boost=sum(0.1 for term in terms if term in text.lower())
        return min(boost,0.5)
class InfluencerWeightingSentiment:
    def __init__(self):
        self.influencer_db={'elonmusk':{'followers':100000000,'crypto_relevance':0.9},'vitalikbuterin':{'followers':5000000,'crypto_relevance':1.0},'michael_saylor':{'followers':2000000,'crypto_relevance':0.95}}
    def calculate_influence_weight(self,author,text):
        if author in self.influencer_db:
            data=self.influencer_db[author];follower_weight=min(np.log10(data['followers'])/8,2.0)
            relevance_weight=data['crypto_relevance'];return follower_weight*relevance_weight
        return 1.0
    def weight_sentiment_by_influence(self,sentiment_data):
        weighted_sentiments=[]
        for item in sentiment_data:
            author=item.get('author','unknown');weight=self.calculate_influence_weight(author,item['text'])
            weighted_sentiment=item['sentiment']*weight
            weighted_sentiments.append({**item,'influence_weight':weight,'weighted_sentiment':weighted_sentiment})
        return weighted_sentiments