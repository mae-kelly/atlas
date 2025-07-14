import torch,torch.nn as nn,torch.nn.functional as F,numpy as np,pandas as pd,logging,yaml,json,random,math,copy,time,hashlib,os
from typing import Dict,List,Tuple,Optional,Any,Union,Callable
from datetime import datetime,timedelta
from dataclasses import dataclass,asdict
from collections import defaultdict,deque,OrderedDict
from sklearn.preprocessing import StandardScaler,LabelEncoder,RobustScaler
from sklearn.model_selection import train_test_split,StratifiedKFold,TimeSeriesSplit
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats,optimize,linalg
from scipy.spatial.distance import pdist,squareform
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt,seaborn as sns
try:import networkx as nx
except:nx=None
try:from sklearn.metrics import mutual_info_score
except:mutual_info_score=None

@dataclass
class AssetSignal:
    ip_address:Optional[str]=None;hostname:Optional[str]=None;mac_address:Optional[str]=None
    source:str="";timestamp:datetime=None;confidence:float=1.0;features:Dict[str,Any]=None;embedding:Optional[torch.Tensor]=None
    def __post_init__(self):
        if self.features is None:self.features={}
        if self.timestamp is None:self.timestamp=datetime.now()

class MultiHeadAttention(nn.Module):
    def __init__(self,d,h=8):
        super().__init__();self.d,self.h,self.dk=d,h,d//h;self.q,self.k,self.v,self.o=nn.Linear(d,d),nn.Linear(d,d),nn.Linear(d,d),nn.Linear(d,d)
    def forward(self,x):
        b,n,d=x.shape;q,k,v=self.q(x).view(b,n,self.h,self.dk).transpose(1,2),self.k(x).view(b,n,self.h,self.dk).transpose(1,2),self.v(x).view(b,n,self.h,self.dk).transpose(1,2)
        s=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.dk);a=F.softmax(s,dim=-1);o=torch.matmul(a,v).transpose(1,2).contiguous().view(b,n,d);return self.o(o)

class ResidualBlock(nn.Module):
    def __init__(self,d):super().__init__();self.l1,self.l2,self.n1,self.n2=nn.Linear(d,d*2),nn.Linear(d*2,d),nn.LayerNorm(d),nn.LayerNorm(d)
    def forward(self,x):r=x;x=self.n1(x);x=F.gelu(self.l1(x));x=F.dropout(x,0.1,self.training);x=self.l2(x);return self.n2(x+r)

class EnhancedAssetModel(nn.Module):
    def __init__(self,input_dim=27,embed_dim=512,num_heads=16,num_layers=8,num_classes=6):
        super().__init__();self.embed=nn.Linear(input_dim,embed_dim);self.attn=nn.ModuleList([MultiHeadAttention(embed_dim,num_heads)for _ in range(num_layers//2)])
        self.res=nn.ModuleList([ResidualBlock(embed_dim)for _ in range(num_layers)]);self.cls=nn.Sequential(nn.Linear(embed_dim,embed_dim//2),nn.GELU(),nn.Dropout(0.1),nn.Linear(embed_dim//2,num_classes))
        self.conf=nn.Sequential(nn.Linear(embed_dim,embed_dim//4),nn.GELU(),nn.Linear(embed_dim//4,1),nn.Sigmoid())
    def forward(self,x):
        x=self.embed(x).unsqueeze(1);[x:=self.res[i](x.squeeze(1)).unsqueeze(1)if i>=len(self.attn)else self.attn[i](x)for i in range(len(self.res))];x=x.squeeze(1)
        return {'predictions':self.cls(x),'confidence':self.conf(x),'embeddings':x}

class AssetCorrelationEngine:
    def __init__(self):self.ip_patterns,self.hostname_patterns={},{};self.correlation_graph=nx.Graph()if nx else None
    def build_correlations(self,datasets):
        correlations={'ip_overlap':{},'hostname_overlap':{},'consistency_matrix':{},'cross_source_overlap':{}}
        all_ips,all_hostnames=set(),set();source_ips,source_hostnames={},{}
        for source,df in datasets.items():
            ips=set(df['ip_address'].dropna()if'ip_address'in df.columns else df['source_ip'].dropna()if'source_ip'in df.columns else[])
            hostnames=set(df['hostname'].dropna().str.lower().str.split('.').str[0]if'hostname'in df.columns else[])
            source_ips[source],source_hostnames[source]=ips,hostnames;all_ips.update(ips);all_hostnames.update(hostnames)
        for s1 in source_ips:
            correlations['ip_overlap'][s1]={}
            for s2 in source_ips:
                i,u=len(source_ips[s1]&source_ips[s2]),len(source_ips[s1]|source_ips[s2])
                correlations['ip_overlap'][s1][s2]=i/u if u>0 else 0
        correlations['cross_source_overlap']={'avg_ip_overlap':np.mean([v for d in correlations['ip_overlap'].values()for k,v in d.items()if k!=list(d.keys())[0]])}
        return correlations

class AtlasNeuralEngine:
    def __init__(self,config_path="config/atlas_config.yaml"):
        self.config=self._load_config(config_path);self.device=torch.device("cpu");self.model,self.correlation_engine=None,AssetCorrelationEngine()
        self.scaler,self.label_encoder=StandardScaler(),LabelEncoder();self.asset_registry,self.confidence_scores={},{}
    def _load_config(self,path):
        try:
            with open(path,'r')as f:return yaml.safe_load(f)
        except:return{'learning_rate':1e-3,'epochs':100,'batch_size':32,'model':{'embed_dim':512,'num_heads':16,'num_layers':8}}
    def extract_features(self,signal):
        f=[]
        if signal.ip_address:
            ip_parts=[int(p)for p in signal.ip_address.split('.')];f.extend([p/255 for p in ip_parts]);f.append(ip_parts[0]/255);f.append((ip_parts[0]==10)or(ip_parts[0]==192 and ip_parts[1]==168))
        else:f.extend([0]*6)
        if signal.hostname:f.extend([len(signal.hostname.split('-')),'prod'in signal.hostname.lower(),'dev'in signal.hostname.lower(),len(signal.hostname)])
        else:f.extend([0]*4)
        source_enc={'chronicle':[1,0,0,0],'crowdstrike':[0,1,0,0],'splunk':[0,0,1,0],'cmdb':[0,0,0,1]};f.extend(source_enc.get(signal.source,[0,0,0,0]))
        if signal.timestamp:f.extend([signal.timestamp.hour/24,signal.timestamp.weekday()/7])
        else:f.extend([0,0])
        f.extend([signal.confidence,len(signal.features)if signal.features else 0]);return np.array(f[:27],dtype=np.float32)
    def cluster_assets(self,signals):
        if len(signals)<2:return{0:list(range(len(signals)))}
        features=np.array([self.extract_features(s)for s in signals]);features_scaled=self.scaler.fit_transform(features)
        clusterer=HDBSCAN(min_cluster_size=2,min_samples=1);labels=clusterer.fit_predict(features_scaled)
        clusters={};[clusters.setdefault(l,[]).append(i)for i,l in enumerate(labels)];return clusters
    def calculate_confidence(self,signal_cluster):
        if not signal_cluster:return 0
        sources=set(s.source for s in signal_cluster);source_diversity=len(sources)/4
        avg_confidence=np.mean([s.confidence for s in signal_cluster]);feature_consistency=1/(1+np.std([len(s.features)for s in signal_cluster]))if len(signal_cluster)>1 else 1
        return min(1,max(0,source_diversity*avg_confidence*feature_consistency))
    def infer_assets(self,signals):
        clusters=self.cluster_assets(signals);inferred_assets=[]
        for cluster_id,signal_indices in clusters.items():
            if cluster_id==-1:continue
            cluster_signals=[signals[i]for i in signal_indices];confidence=self.calculate_confidence(cluster_signals)
            asset={'asset_id':f"atlas_{hash(str(sorted([s.ip_address for s in cluster_signals if s.ip_address])))%100000:05d}",'signals':cluster_signals,'confidence':confidence,'ip_addresses':list(set(s.ip_address for s in cluster_signals if s.ip_address)),'hostnames':list(set(s.hostname for s in cluster_signals if s.hostname)),'sources':list(set(s.source for s in cluster_signals))}
            inferred_assets.append(asset)
        return{'inferred_assets':inferred_assets,'shadow_assets':[a for a in inferred_assets if'cmdb'not in[s.source for s in a['signals']]],'visibility_score':min(1,sum(a['confidence']for a in inferred_assets)/len(inferred_assets))if inferred_assets else 0}

def generate_enhanced_datasets(scale_factor=1):
    base_size=10000*scale_factor;shared_ips=[f"10.0.{i//254}.{i%254+1}"for i in range(2000)];shared_hostnames=[f"asset-{i:04d}.corp.local"for i in range(1000)];datasets={}
    def create_chronicle_data():
        data=[]
        for i in range(base_size):
            ip=np.random.choice(shared_ips)if np.random.random()<0.8 else f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
            data.append({'source_ip':ip,'source_port':np.random.choice([22,80,443,3389,8080]),'protocol':np.random.choice(['TCP','UDP']),'bytes_sent':np.random.randint(1000,100000),'bytes_received':np.random.randint(1000,100000),'asset_type':np.random.choice(['server','workstation','iot_device','mobile_device','network_device','cloud_service']),'confidence_score':np.random.beta(8,2)})
        return pd.DataFrame(data)
    def create_splunk_data():
        data=[]
        for i in range(int(base_size*0.7)):
            hostname=np.random.choice(shared_hostnames)if np.random.random()<0.6 else f"host-{np.random.randint(1000,9999)}"
            data.append({'hostname':hostname,'ip_address':np.random.choice(shared_ips),'cpu_usage':np.random.uniform(10,90),'memory_usage':np.random.uniform(20,85),'asset_type':np.random.choice(['server','workstation','iot_device','mobile_device','network_device','cloud_service'])})
        return pd.DataFrame(data)
    def create_crowdstrike_data():
        data=[]
        for i in range(int(base_size*0.5)):
            data.append({'endpoint_id':f"CS-{np.random.randint(100000,999999)}",'hostname':np.random.choice(shared_hostnames[:len(shared_hostnames)//2])if np.random.random()<0.4 else f"endpoint-{i}",'ip_address':np.random.choice(shared_ips),'threat_detections':np.random.randint(0,10),'risk_score':np.random.uniform(10,70),'asset_type':np.random.choice(['server','workstation','iot_device','mobile_device','network_device','cloud_service'])})
        return pd.DataFrame(data)
    def create_cmdb_data():
        data=[]
        for i in range(int(base_size*0.3)):
            data.append({'ci_id':f"CI{np.random.randint(100000,999999)}",'hostname':np.random.choice(shared_hostnames)if np.random.random()<0.9 else f"cmdb-{i}",'ip_address':np.random.choice(shared_ips),'asset_status':'active','asset_type':np.random.choice(['server','workstation','iot_device','mobile_device','network_device','cloud_service'])})
        return pd.DataFrame(data)
    datasets['chronicle'],datasets['splunk'],datasets['crowdstrike'],datasets['cmdb']=create_chronicle_data(),create_splunk_data(),create_crowdstrike_data(),create_cmdb_data();return datasets

class EnhancedTrainingPipeline:
    def __init__(self,config):self.config,self.device=config,torch.device("cpu");self.model,self.optimizer,self.criterion=None,None,None;self.scaler,self.label_encoder=StandardScaler(),LabelEncoder()
    def prepare_data(self,datasets):
        features,labels=[],[]
        for source,df in datasets.items():
            for _,row in df.iterrows():
                signal=AssetSignal(ip_address=row.get('ip_address'or'source_ip'),hostname=row.get('hostname'),source=source,confidence=row.get('confidence_score',0.8));engine=AtlasNeuralEngine();f=engine.extract_features(signal);features.append(f);labels.append(row.get('asset_type','unknown'))
        X,y=np.array(features),self.label_encoder.fit_transform(labels);X=self.scaler.fit_transform(X);X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42,stratify=y_train)
        return{k:torch.FloatTensor(v)if'X'in k else torch.LongTensor(v)for k,v in{'X_train':X_train,'X_val':X_val,'X_test':X_test,'y_train':y_train,'y_val':y_val,'y_test':y_test}.items()}
    def build_model(self,input_dim,num_classes):
        self.model=EnhancedAssetModel(input_dim,num_classes=num_classes).to(self.device);self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.config.get('learning_rate',1e-3),weight_decay=0.01);self.criterion=nn.CrossEntropyLoss();return self.model
    def train_model(self,data_tensors,epochs=100):
        train_loader=[(data_tensors['X_train'][i:i+32],data_tensors['y_train'][i:i+32])for i in range(0,len(data_tensors['X_train']),32)]
        val_loader=[(data_tensors['X_val'][i:i+64],data_tensors['y_val'][i:i+64])for i in range(0,len(data_tensors['X_val']),64)]
        scheduler=torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=1e-3,epochs=epochs,steps_per_epoch=len(train_loader))
        best_val_acc,patience,patience_counter=0,15,0
        for epoch in range(epochs):
            self.model.train();total_loss,correct,total=0,0,0
            for batch_x,batch_y in train_loader:
                self.optimizer.zero_grad();outputs=self.model(batch_x);loss=self.criterion(outputs['predictions'],batch_y);loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);self.optimizer.step();scheduler.step()
                total_loss+=loss.item();correct+=(outputs['predictions'].argmax(1)==batch_y).sum().item();total+=batch_y.size(0)
            train_acc=correct/total;self.model.eval();val_correct,val_total=0,0
            with torch.no_grad():
                for batch_x,batch_y in val_loader:outputs=self.model(batch_x);val_correct+=(outputs['predictions'].argmax(1)==batch_y).sum().item();val_total+=batch_y.size(0)
            val_acc=val_correct/val_total
            if val_acc>best_val_acc:best_val_acc=val_acc;patience_counter=0;torch.save(self.model.state_dict(),'best_model.pth')
            else:patience_counter+=1
            if epoch%10==0:print(f"Epoch {epoch:3d}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
            if patience_counter>=patience:print(f"Early stopping at epoch {epoch}");break
        self.model.load_state_dict(torch.load('best_model.pth'));return{'best_val_accuracy':best_val_acc}
    def evaluate_model(self,data_tensors):
        self.model.eval();X_test,y_test=data_tensors['X_test'],data_tensors['y_test']
        with torch.no_grad():outputs=self.model(X_test);pred_classes=outputs['predictions'].argmax(1).cpu().numpy();true_classes=y_test.cpu().numpy();confidence_scores=outputs['confidence'].cpu().numpy()
        accuracy=accuracy_score(true_classes,pred_classes);precision,recall,f1,_=precision_recall_fscore_support(true_classes,pred_classes,average='weighted')
        return{'test_accuracy':accuracy,'precision':precision,'recall':recall,'f1_score':f1,'confidence_scores':confidence_scores,'predictions':pred_classes,'true_labels':true_classes}

def main():
    print("🚀 Atlas Neural Asset Discovery Engine - Compressed Version");print("Target: 95%+ Asset Discovery, 85%+ Cross-Source Consistency")
    config={'learning_rate':3e-4,'epochs':50,'batch_size':32};datasets=generate_enhanced_datasets(scale_factor=1);pipeline=EnhancedTrainingPipeline(config)
    print(f"📊 Generated {sum(len(df)for df in datasets.values()):,} records");data_tensors=pipeline.prepare_data(datasets);input_dim,num_classes=data_tensors['X_train'].shape[1],len(torch.unique(data_tensors['y_train']))
    model=pipeline.build_model(input_dim,num_classes);print(f"🧠 Model: {input_dim}→512D, {sum(p.numel()for p in model.parameters()):,} params")
    training_results=pipeline.train_model(data_tensors,epochs=config['epochs']);evaluation_results=pipeline.evaluate_model(data_tensors)
    test_accuracy,precision,recall,f1=evaluation_results['test_accuracy'],evaluation_results['precision'],evaluation_results['recall'],evaluation_results['f1_score']
    print(f"\n🏆 RESULTS:\n  Asset Discovery Accuracy: {test_accuracy:.3f}\n  Precision: {precision:.3f}\n  Recall: {recall:.3f}\n  F1-Score: {f1:.3f}")
    correlation_engine=AssetCorrelationEngine();signals=[AssetSignal(ip_address=row.get('ip_address'or'source_ip'),hostname=row.get('hostname'),source=source)for source,df in datasets.items()for _,row in df.head(100).iterrows()]
    correlations=correlation_engine.build_correlations(datasets);consistency=correlations['cross_source_overlap']['avg_ip_overlap']
    print(f"  Cross-Source Consistency: {consistency:.3f}\n")
    print(f"🎯 TARGET ACHIEVEMENT:")
    print(f"   Asset Discovery ≥95%: {'✅ ACHIEVED'if test_accuracy>=0.95 else'❌ MISSED'}")
    print(f"   Cross-Source ≥85%: {'✅ ACHIEVED'if consistency>=0.85 else'❌ MISSED'}")
    if test_accuracy>=0.95 and consistency>=0.85:print("🏆 SUCCESS! Both targets achieved!")
    results={'timestamp':datetime.now().isoformat(),'test_accuracy':test_accuracy,'cross_source_consistency':consistency,'training_results':training_results,'config':config}
    with open('results/atlas_results.json','w')as f:json.dump(results,f,indent=2,default=str)
    return results

if __name__=="__main__":main()
