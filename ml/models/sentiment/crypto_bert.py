import torch,torch.nn as nn,transformers,numpy as np,pandas as pd
from transformers import AutoTokenizer,AutoModel,pipeline
from typing import Dict,List,Tuple
import re

class CryptoBERT:
    def __init__(self,model_name:str='nlptown/bert-base-multilingual-uncased-sentiment'):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModel.from_pretrained(model_name)
        self.sentiment_pipeline=pipeline('sentiment-analysis',model=model_name,tokenizer=model_name)
        self.crypto_vocab={'bitcoin':0.1,'btc':0.1,'ethereum':0.08,'eth':0.08,'moon':0.15,'hodl':0.12,'dump':-0.15,'crash':-0.2,'bull':0.1,'bear':-0.1,'fud':-0.12,'fomo':0.08,'rekt':-0.18,'diamond_hands':0.1,'paper_hands':-0.08,'ath':0.12,'dip':-0.05,'pump':0.15}
    
    def analyze_crypto_sentiment(self,texts:List[str])->List[Dict]:
        results=[]
        for text in texts:
            cleaned_text=self._preprocess_crypto_text(text)
            base_sentiment=self.sentiment_pipeline(cleaned_text)[0]
            crypto_boost=self._calculate_crypto_sentiment_boost(text)
            emotion_scores=self._analyze_emotions(text)
            final_score=self._combine_sentiment_scores(base_sentiment,crypto_boost,emotion_scores)
            results.append({'text':text,'base_sentiment':base_sentiment,'crypto_sentiment':crypto_boost,'emotion_scores':emotion_scores,'final_sentiment':final_score,'confidence':final_score.get('confidence',0.5)})
        return results
    
    def _preprocess_crypto_text(self,text:str)->str:
        text=re.sub(r'http\S+|www\S+|https\S+','',text,flags=re.MULTILINE)
        text=re.sub(r'@\w+|#\w+','',text);text=re.sub(r'[^\w\s]',' ',text);return' '.join(text.split())
    
    def _calculate_crypto_sentiment_boost(self,text:str)->Dict[str,float]:
        text_lower=text.lower();boost_score,term_count=0,0
        for term,weight in self.crypto_vocab.items():
            if term in text_lower:boost_score+=weight;term_count+=1
        return{'boost_score':boost_score,'terms_found':term_count,'normalized_boost':boost_score/max(term_count,1)}
    
    def _analyze_emotions(self,text:str)->Dict[str,float]:
        emotion_patterns={'excitement':['amazing','incredible','moon','rocket','fire','explosive'],'fear':['crash','dump','panic','worried','scared','disaster'],'greed':['profit','gains','money','rich','wealth','lambo'],'uncertainty':['maybe','unsure','confused','unclear','doubt','risky']}
        emotion_scores={}
        for emotion,patterns in emotion_patterns.items():
            score=sum(1 for pattern in patterns if pattern in text.lower())
            emotion_scores[emotion]=min(score/len(patterns),1.0)
        return emotion_scores
    
    def _combine_sentiment_scores(self,base_sentiment:Dict,crypto_boost:Dict,emotions:Dict)->Dict:
        base_score=1 if base_sentiment['label']=='POSITIVE'else-1
        base_confidence=base_sentiment['score']
        crypto_adjustment=crypto_boost['normalized_boost']
        emotion_adjustment=(emotions.get('excitement',0)-emotions.get('fear',0)+emotions.get('greed',0)*0.5-emotions.get('uncertainty',0)*0.3)
        final_score=base_score*base_confidence+crypto_adjustment+emotion_adjustment*0.3
        final_confidence=min(base_confidence+abs(crypto_adjustment)*0.5+max(emotions.values())*0.2,1.0)
        return{'sentiment_score':np.clip(final_score,-1,1),'confidence':final_confidence,'label':'bullish'if final_score>0.1 else'bearish'if final_score<-0.1 else'neutral'}

class AspectLevelSentiment:
    def __init__(self):
        self.aspects={'technology':['blockchain','protocol','upgrade','development','code','technical'],'adoption':['partnership','integration','mainstream','institution','corporate','government'],'regulation':['sec','regulatory','legal','compliance','ban','approval'],'market':['price','trading','volume','market','exchange','liquidity'],'community':['community','social','reddit','twitter','telegram','discord']}
        self.aspect_models={aspect:pipeline('sentiment-analysis',model='cardiffnlp/twitter-roberta-base-sentiment-latest')for aspect in self.aspects}
    
    def analyze_aspects(self,text:str)->Dict[str,Dict]:
        results={}
        for aspect,keywords in self.aspects.items():
            aspect_text=self._extract_aspect_text(text,keywords)
            if aspect_text:
                sentiment=self.aspect_models[aspect](aspect_text)[0]
                results[aspect]={'sentiment':sentiment,'text_snippet':aspect_text[:100],'relevance':len([k for k in keywords if k in text.lower()])}
            else:results[aspect]={'sentiment':{'label':'NEUTRAL','score':0.5},'text_snippet':'','relevance':0}
        return results
    
    def _extract_aspect_text(self,text:str,keywords:List[str])->str:
        sentences=[s.strip()for s in text.split('.')if s.strip()]
        relevant_sentences=[s for s in sentences if any(keyword in s.lower()for keyword in keywords)]
        return' '.join(relevant_sentences[:2])

class MultilingualSentiment:
    def __init__(self):
        self.language_models={'en':'cardiffnlp/twitter-roberta-base-sentiment-latest','es':'cardiffnlp/twitter-xlm-roberta-base-sentiment','fr':'cardiffnlp/twitter-xlm-roberta-base-sentiment','de':'cardiffnlp/twitter-xlm-roberta-base-sentiment','ja':'cardiffnlp/twitter-xlm-roberta-base-sentiment','ko':'cardiffnlp/twitter-xlm-roberta-base-sentiment','zh':'cardiffnlp/twitter-xlm-roberta-base-sentiment'}
        self.pipelines={lang:pipeline('sentiment-analysis',model=model)for lang,model in self.language_models.items()}
    
    def detect_language(self,text:str)->str:
        char_patterns={'en':r'[a-zA-Z]','es':r'[ñáéíóúü]','fr':r'[àâäéèêëïîôöùûüÿç]','de':r'[äöüß]','ja':r'[ひらがなカタカナ]','ko':r'[ㄱ-ㅎㅏ-ㅣ가-힣]','zh':r'[\u4e00-\u9fff]'}
        scores={lang:len(re.findall(pattern,text.lower()))for lang,pattern in char_patterns.items()}
        return max(scores,key=scores.get)if max(scores.values())>0 else'en'
    
    def analyze_multilingual(self,text:str)->Dict:
        detected_lang=self.detect_language(text)
        pipeline_lang=detected_lang if detected_lang in self.pipelines else'en'
        sentiment=self.pipelines[pipeline_lang](text)[0]
        return{'detected_language':detected_lang,'sentiment':sentiment,'pipeline_used':pipeline_lang}

class InfluencerWeightedSentiment:
    def __init__(self):
        self.influencer_weights={'elon_musk':10.0,'vitalik_buterin':8.0,'cz_binance':7.0,'michael_saylor':6.0,'cathie_wood':5.0,'coindesk':4.0,'cointelegraph':3.0,'default':1.0}
    
    def weight_by_influence(self,sentiment_data:List[Dict])->List[Dict]:
        weighted_results=[]
        for item in sentiment_data:
            author=item.get('author','default').lower()
            follower_count=item.get('follower_count',1000)
            base_weight=self.influencer_weights.get(author,self.influencer_weights['default'])
            follower_weight=min(np.log10(follower_count)/6,2.0)
            total_weight=base_weight*follower_weight
            weighted_sentiment=item['sentiment_score']*total_weight
            weighted_results.append({**item,'influence_weight':total_weight,'weighted_sentiment':weighted_sentiment})
        return weighted_results
