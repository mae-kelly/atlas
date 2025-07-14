import torch,torch.nn as nn,networkx as nx,numpy as np
from typing import Dict,List,Tuple
class DynamicKnowledgeGraph:
    def __init__(self):
        self.G=nx.DiGraph();self.entity_emb=nn.Embedding(10000,256);self.relation_emb=nn.Embedding(100,256)
        self.update_net=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,256))
    def add_entity(self,entity,features):self.G.add_node(entity,**features)
    def add_relation(self,head,tail,relation,weight=1.0):self.G.add_edge(head,tail,relation=relation,weight=weight)
    def update_embeddings(self,news_text):
        entities=self.extract_entities(news_text);relations=self.extract_relations(news_text)
        for e in entities:self.entity_emb.weight.data[hash(e)%10000]+=torch.randn(256)*0.01
    def extract_entities(self,text):return [w for w in text.split()if w.isupper()]
    def extract_relations(self,text):return ['AFFECTS','CORRELATES']
    def query_graph(self,entity):return list(self.G.neighbors(entity))if entity in self.G else[]
class SpatialTemporalAttention(nn.Module):
    def __init__(self,dim,num_nodes):
        super().__init__();self.spatial_attn=nn.MultiheadAttention(dim,8,batch_first=True)
        self.temporal_attn=nn.MultiheadAttention(dim,8,batch_first=True);self.node_emb=nn.Embedding(num_nodes,dim)
    def forward(self,x,node_ids,timestamps):
        spatial_x,_=self.spatial_attn(x,x,x);temporal_x,_=self.temporal_attn(spatial_x,spatial_x,spatial_x)
        node_features=self.node_emb(node_ids);return temporal_x+node_features
class CausalMemoryNetwork(nn.Module):
    def __init__(self,dim):
        super().__init__();self.memory=nn.Parameter(torch.randn(100,dim));self.causal_net=nn.Linear(dim*2,1)
        self.update_gate=nn.Sequential(nn.Linear(dim*2,dim),nn.Sigmoid())
    def forward(self,x,sentiment):
        memory_scores=torch.mm(x,self.memory.T).softmax(-1);retrieved=torch.mm(memory_scores,self.memory)
        causal_score=self.causal_net(torch.cat([x,sentiment],dim=-1)).sigmoid()
        gate=self.update_gate(torch.cat([x,retrieved],dim=-1));updated_memory=gate*retrieved+(1-gate)*x
        return updated_memory*causal_score
class HierarchicalGraphAttention(nn.Module):
    def __init__(self,company_dim,industry_dim):
        super().__init__();self.company_attn=nn.MultiheadAttention(company_dim,4,batch_first=True)
        self.industry_attn=nn.MultiheadAttention(industry_dim,4,batch_first=True)
        self.cross_attn=nn.MultiheadAttention(company_dim,4,batch_first=True)
    def forward(self,company_features,industry_features):
        company_out,_=self.company_attn(company_features,company_features,company_features)
        industry_out,_=self.industry_attn(industry_features,industry_features,industry_features)
        cross_out,_=self.cross_attn(company_out,industry_out,industry_out);return cross_out+company_out