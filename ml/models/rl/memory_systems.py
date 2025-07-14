import torch,torch.nn as nn

class FinMemFramework(nn.Module):
    def __init__(self,input_dim,memory_size=1000):
        super().__init__();self.memory_size=memory_size
        self.profiling_net=nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),nn.Linear(128,64))
        self.memory_bank=nn.Parameter(torch.randn(memory_size,64))
        self.decision_net=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.update_gate=nn.Sequential(nn.Linear(128,memory_size),nn.Sigmoid())
    def forward(self,x):
        profile=self.profiling_net(x);similarity=torch.mm(profile,self.memory_bank.T).softmax(-1)
        retrieved=torch.mm(similarity,self.memory_bank);combined=torch.cat([profile,retrieved],dim=1)
        decision=self.decision_net(combined);update_weights=self.update_gate(combined)
        self.memory_bank.data=update_weights.T@profile+(1-update_weights).T@self.memory_bank.data
        return decision

class MultiAgentSystem(nn.Module):
    def __init__(self,state_dim):
        super().__init__()
        self.fundamental_agent=nn.Sequential(nn.Linear(state_dim,64),nn.ReLU(),nn.Linear(64,1))
        self.sentiment_agent=nn.Sequential(nn.Linear(state_dim,64),nn.ReLU(),nn.Linear(64,1))
        self.technical_agent=nn.Sequential(nn.Linear(state_dim,64),nn.ReLU(),nn.Linear(64,1))
        self.risk_agent=nn.Sequential(nn.Linear(state_dim,64),nn.ReLU(),nn.Linear(64,1))
        self.coordinator=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
    def forward(self,x):
        fundamental=self.fundamental_agent(x);sentiment=self.sentiment_agent(x)
        technical=self.technical_agent(x);risk=self.risk_agent(x)
        combined=torch.cat([fundamental,sentiment,technical,risk],dim=1)
        return self.coordinator(combined)

class AlphaGPT(nn.Module):
    def __init__(self,vocab_size,dim=512):
        super().__init__();self.emb=nn.Embedding(vocab_size,dim)
        self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,8,dim*4,0.1,batch_first=True),6)
        self.factor_head=nn.Linear(dim,1);self.code_head=nn.Linear(dim,vocab_size)
    def forward(self,x):
        x=self.emb(x);x=self.transformer(x);factor=self.factor_head(x[:,-1]);code=self.code_head(x)
        return{'alpha_factor':factor,'generated_code':code}
