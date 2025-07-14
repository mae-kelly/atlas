import torch,networkx as nx,pandas as pd,numpy as np
from torch_geometric.data import Data
from .recurrent_gnn import RecurrentGraphNeuralNetwork,SGATBootstrapModel
from typing import Dict,List,Tuple
from loguru import logger

class BlockchainTransactionAnalyzer:
    def __init__(self):
        self.rec_gnn=RecurrentGraphNeuralNetwork(node_features=64,hidden_dim=128,num_layers=3,num_classes=2)
        self.sgat_model=SGATBootstrapModel(input_dim=64,hidden_dim=128,num_heads=8,num_layers=3)
        self.device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
        self.rec_gnn.to(self.device);self.sgat_model.to(self.device)
    
    def analyze_transaction_graph(self,transactions:List[Dict])->Dict[str,float]:
        graph_data=self._build_transaction_graph(transactions)
        illicit_probability=self._detect_illicit_activity(graph_data)
        community_analysis=self._analyze_communities(graph_data)
        risk_score=self._calculate_risk_score(illicit_probability,community_analysis)
        return{'illicit_probability':illicit_probability,'community_analysis':community_analysis,'risk_score':risk_score,'total_transactions':len(transactions)}
    
    def _build_transaction_graph(self,transactions:List[Dict])->Data:
        G=nx.DiGraph()
        for tx in transactions:
            G.add_edge(tx['from'],tx['to'],weight=tx['value'],timestamp=tx['timestamp'])
        nodes=list(G.nodes());node_mapping={node:i for i,node in enumerate(nodes)}
        edge_index=torch.tensor([[node_mapping[edge[0]],node_mapping[edge[1]]]for edge in G.edges()]).t()
        node_features=torch.randn(len(nodes),64)
        for i,node in enumerate(nodes):
            in_degree,out_degree=G.in_degree(node),G.out_degree(node)
            total_value_in=sum([G[u][node].get('weight',0)for u in G.predecessors(node)])
            total_value_out=sum([G[node][v].get('weight',0)for v in G.successors(node)])
            node_features[i,:4]=torch.tensor([in_degree,out_degree,total_value_in,total_value_out])
        return Data(x=node_features,edge_index=edge_index)
    
    def _detect_illicit_activity(self,graph_data:Data)->float:
        self.rec_gnn.eval()
        with torch.no_grad():
            graph_data=graph_data.to(self.device);predictions=self.rec_gnn([graph_data])
            illicit_prob=torch.softmax(predictions,dim=1)[0,1].item()
        return illicit_prob
    
    def _analyze_communities(self,graph_data:Data)->Dict:
        edge_index=graph_data.edge_index.cpu().numpy()
        G=nx.Graph();G.add_edges_from(edge_index.T)
        try:
            communities=nx.community.louvain_communities(G)
            community_info={'num_communities':len(communities),'largest_community':max(len(c)for c in communities)if communities else 0,'modularity':nx.community.modularity(G,communities)if communities else 0}
        except:community_info={'num_communities':0,'largest_community':0,'modularity':0}
        return community_info
    
    def _calculate_risk_score(self,illicit_prob:float,community_analysis:Dict)->float:
        base_risk=illicit_prob*0.7
        community_risk=(1-community_analysis['modularity'])*0.2 if community_analysis['modularity']>0 else 0.2
        size_risk=min(community_analysis['largest_community']/100,0.1)
        return min(base_risk+community_risk+size_risk,1.0)

class DeFiProtocolAnalyzer:
    def __init__(self):
        self.risk_patterns={'flash_loan_attack':0.9,'reentrancy':0.95,'oracle_manipulation':0.85,'governance_attack':0.8}
    
    def analyze_defi_protocol(self,protocol_data:Dict)->Dict:
        transactions=protocol_data.get('transactions',[]);liquidity_events=protocol_data.get('liquidity_events',[])
        governance_events=protocol_data.get('governance_events',[]);oracle_updates=protocol_data.get('oracle_updates',[])
        risk_assessment=self._assess_protocol_risk(transactions,liquidity_events,governance_events,oracle_updates)
        tvl_analysis=self._analyze_tvl_changes(liquidity_events)
        return{'risk_assessment':risk_assessment,'tvl_analysis':tvl_analysis,'total_transactions':len(transactions)}
    
    def _assess_protocol_risk(self,transactions:List,liquidity_events:List,governance_events:List,oracle_updates:List)->Dict:
        risks={}
        if self._detect_flash_loan_pattern(transactions):risks['flash_loan_attack']=self.risk_patterns['flash_loan_attack']
        if self._detect_reentrancy_pattern(transactions):risks['reentrancy']=self.risk_patterns['reentrancy']
        if self._detect_oracle_manipulation(oracle_updates):risks['oracle_manipulation']=self.risk_patterns['oracle_manipulation']
        if self._detect_governance_attack(governance_events):risks['governance_attack']=self.risk_patterns['governance_attack']
        return risks
    
    def _detect_flash_loan_pattern(self,transactions:List)->bool:
        return any(tx.get('type')=='flash_loan'for tx in transactions)
    def _detect_reentrancy_pattern(self,transactions:List)->bool:
        return len([tx for tx in transactions if tx.get('gas_used',0)>500000])>5
    def _detect_oracle_manipulation(self,oracle_updates:List)->bool:
        if len(oracle_updates)<2:return False
        price_changes=[abs(oracle_updates[i]['price']-oracle_updates[i-1]['price'])/oracle_updates[i-1]['price']for i in range(1,len(oracle_updates))]
        return any(change>0.1 for change in price_changes)
    def _detect_governance_attack(self,governance_events:List)->bool:
        return any(event.get('proposal_type')=='emergency'for event in governance_events)
    def _analyze_tvl_changes(self,liquidity_events:List)->Dict:
        if not liquidity_events:return{'tvl_change':0,'volatility':0}
        tvl_values=[event['tvl']for event in liquidity_events]
        tvl_change=(tvl_values[-1]-tvl_values[0])/tvl_values[0]if len(tvl_values)>1 else 0
        volatility=np.std(tvl_values)/np.mean(tvl_values)if tvl_values else 0
        return{'tvl_change':tvl_change,'volatility':volatility}
