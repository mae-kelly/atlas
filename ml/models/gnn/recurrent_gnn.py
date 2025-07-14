import torch,torch.nn as nn,torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv,GATConv,global_mean_pool
from torch_geometric.data import Data,Batch
import numpy as np
from typing import Dict,List,Tuple

class RecurrentGraphNeuralNetwork(nn.Module):
    def __init__(self,node_features:int=64,hidden_dim:int=128,num_layers:int=3,num_classes:int=2):
        super().__init__()
        self.node_features,self.hidden_dim,self.num_layers=node_features,hidden_dim,num_layers
        self.node_embedding=nn.Linear(node_features,hidden_dim)
        self.gnn_layers=nn.ModuleList([GCNConv(hidden_dim,hidden_dim)for _ in range(num_layers)])
        self.lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.attention=nn.MultiheadAttention(hidden_dim,num_heads=8,batch_first=True)
        self.classifier=nn.Sequential(nn.Linear(hidden_dim,hidden_dim//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hidden_dim//2,num_classes))
    
    def forward(self,batch_graphs:List[Data])->torch.Tensor:
        batch=Batch.from_data_list(batch_graphs);x,edge_index,batch_idx=batch.x,batch.edge_index,batch.batch
        x=self.node_embedding(x)
        for gnn_layer in self.gnn_layers:x=F.relu(gnn_layer(x,edge_index))
        graph_embeddings=global_mean_pool(x,batch_idx)
        if graph_embeddings.dim()==2:graph_embeddings=graph_embeddings.unsqueeze(1)
        lstm_out,(h_n,c_n)=self.lstm(graph_embeddings);attended_out,_=self.attention(lstm_out,lstm_out,lstm_out)
        final_embedding=attended_out.mean(dim=1);return self.classifier(final_embedding)

class SGATBootstrapModel(nn.Module):
    def __init__(self,input_dim:int=64,hidden_dim:int=128,num_heads:int=8,num_layers:int=3):
        super().__init__()
        self.gat_layers=nn.ModuleList([GATConv(input_dim if i==0 else hidden_dim*num_heads,hidden_dim,heads=num_heads,dropout=0.3)for i in range(num_layers)])
        self.bootstrap_layers=nn.ModuleList([nn.Linear(hidden_dim*num_heads,hidden_dim*num_heads)for _ in range(num_layers)])
        self.classifier=nn.Sequential(nn.Linear(hidden_dim*num_heads,hidden_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden_dim,2))
    
    def forward(self,x:torch.Tensor,edge_index:torch.Tensor,batch:torch.Tensor)->torch.Tensor:
        for i,(gat_layer,bootstrap_layer)in enumerate(zip(self.gat_layers,self.bootstrap_layers)):
            x=F.elu(gat_layer(x,edge_index))
            if i<len(self.gat_layers)-1:
                bootstrap_noise=torch.randn_like(x)*0.1;x=bootstrap_layer(x+bootstrap_noise)
        graph_embeddings=global_mean_pool(x,batch);return self.classifier(graph_embeddings)

class HypergraphNN(nn.Module):
    def __init__(self,node_features:int,hyperedge_features:int,hidden_dim:int=128):
        super().__init__()
        self.node_proj=nn.Linear(node_features,hidden_dim);self.hyperedge_proj=nn.Linear(hyperedge_features,hidden_dim)
        self.node_hyperedge_conv=nn.Linear(hidden_dim,hidden_dim);self.hyperedge_node_conv=nn.Linear(hidden_dim,hidden_dim)
        self.output_proj=nn.Linear(hidden_dim,2)
    
    def forward(self,node_features:torch.Tensor,hyperedge_features:torch.Tensor,incidence_matrix:torch.Tensor)->torch.Tensor:
        node_emb=self.node_proj(node_features);hyperedge_emb=self.hyperedge_proj(hyperedge_features)
        for _ in range(3):
            node_to_hyperedge=torch.mm(incidence_matrix.t(),node_emb);hyperedge_emb=hyperedge_emb+self.node_hyperedge_conv(node_to_hyperedge)
            hyperedge_to_node=torch.mm(incidence_matrix,hyperedge_emb);node_emb=node_emb+self.hyperedge_node_conv(hyperedge_to_node)
        return self.output_proj(node_emb.mean(dim=0,keepdim=True))
