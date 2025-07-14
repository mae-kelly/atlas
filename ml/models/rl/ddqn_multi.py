import torch,torch.nn as nn,torch.nn.functional as F,numpy as np,random
from collections import deque

class DDQN_MultiArch(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.lstm=nn.LSTM(state_dim,128,batch_first=True)
        self.bilstm=nn.LSTM(state_dim,64,batch_first=True,bidirectional=True)
        self.gru=nn.GRU(state_dim,64,batch_first=True)
        self.fc=nn.Sequential(nn.Linear(128+128+64,256),nn.ReLU(),nn.Linear(256,action_dim))
        self.value=nn.Linear(256,1);self.advantage=nn.Linear(256,action_dim)
    def forward(self,x):
        lstm_out,_=self.lstm(x);bilstm_out,_=self.bilstm(x);gru_out,_=self.gru(x)
        combined=torch.cat([lstm_out[:,-1],bilstm_out[:,-1],gru_out[:,-1]],dim=1)
        features=F.relu(self.fc[0](combined));value=self.value(features);advantage=self.advantage(features)
        return value+advantage-advantage.mean(dim=1,keepdim=True)

class EnsembleRL(nn.Module):
    def __init__(self,state_dim,action_dim,n_agents=5):
        super().__init__();self.agents=nn.ModuleList([DDQN_MultiArch(state_dim,action_dim)for _ in range(n_agents)])
        self.weights=nn.Parameter(torch.ones(n_agents)/n_agents)
    def forward(self,x):
        outputs=[agent(x)for agent in self.agents];weighted=torch.stack(outputs)*self.weights.view(-1,1,1)
        return weighted.sum(0)

class CVaR_RL(nn.Module):
    def __init__(self,state_dim,action_dim,alpha=0.05):
        super().__init__();self.alpha=alpha;self.net=DDQN_MultiArch(state_dim,action_dim)
        self.risk_net=nn.Sequential(nn.Linear(state_dim,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):
        q_values=self.net(x);risk_scores=self.risk_net(x[:,-1]);cvar_penalty=risk_scores*self.alpha
        return q_values-cvar_penalty.unsqueeze(1)

class MetaLearningRL(nn.Module):
    def __init__(self,state_dim,action_dim,n_tasks=10):
        super().__init__();self.meta_net=nn.LSTM(state_dim+action_dim+1,128,batch_first=True)
        self.task_nets=nn.ModuleList([DDQN_MultiArch(state_dim,action_dim)for _ in range(n_tasks)])
        self.task_selector=nn.Sequential(nn.Linear(128,n_tasks),nn.Softmax(dim=1))
    def forward(self,x,task_history):
        meta_out,_=self.meta_net(task_history);task_probs=self.task_selector(meta_out[:,-1])
        task_outputs=torch.stack([net(x)for net in self.task_nets]);return(task_outputs*task_probs.unsqueeze(-1)).sum(0)
