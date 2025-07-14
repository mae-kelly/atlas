import torch,torch.nn as nn,numpy as np,random
from typing import List,Callable

class SymbolicNode:
    def __init__(self,op,left=None,right=None,value=None):
        self.op,self.left,self.right,self.value=op,left,right,value
    def evaluate(self,x):
        if self.value is not None:return self.value if isinstance(self.value,float)else x[self.value]
        if self.op=='+':return self.left.evaluate(x)+self.right.evaluate(x)
        elif self.op=='*':return self.left.evaluate(x)*self.right.evaluate(x)
        elif self.op=='sin':return np.sin(self.left.evaluate(x))
        elif self.op=='log':return np.log(abs(self.left.evaluate(x))+1e-8)
        return 0

class SGP_Generator:
    def __init__(self,input_vars):
        self.input_vars=input_vars;self.ops=['+','*','sin','log']
    def generate_tree(self,max_depth=3):
        if max_depth==0 or random.random()<0.3:
            return SymbolicNode(None,value=random.choice([random.random()*2-1]+list(range(len(self.input_vars)))))
        op=random.choice(self.ops);left=self.generate_tree(max_depth-1)
        right=self.generate_tree(max_depth-1)if op in['+','*']else None
        return SymbolicNode(op,left,right)

class SGP_LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim=128,n_programs=10):
        super().__init__();self.sgp=SGP_Generator(list(range(input_dim)));self.programs=[self.sgp.generate_tree()for _ in range(n_programs)]
        self.lstm=nn.LSTM(input_dim+n_programs,hidden_dim,batch_first=True)
        self.output=nn.Linear(hidden_dim,1);self.program_weights=nn.Parameter(torch.randn(n_programs))
    def forward(self,x):
        b,s,d=x.shape;sgp_features=torch.zeros(b,s,len(self.programs),device=x.device)
        for i,prog in enumerate(self.programs):
            for j in range(b):
                for k in range(s):
                    try:sgp_features[j,k,i]=prog.evaluate(x[j,k].cpu().numpy())
                    except:sgp_features[j,k,i]=0
        weighted_sgp=(sgp_features*self.program_weights).sum(-1,keepdim=True)
        combined=torch.cat([x,weighted_sgp],dim=-1);lstm_out,_=self.lstm(combined)
        return self.output(lstm_out[:,-1])

class CNN_LSTM_GRU(nn.Module):
    def __init__(self,input_dim,seq_len):
        super().__init__()
        self.conv1d=nn.Conv1d(input_dim,64,3,padding=1);self.lstm=nn.LSTM(64,128,batch_first=True)
        self.gru=nn.GRU(128,64,batch_first=True);self.output=nn.Linear(64,1)
    def forward(self,x):
        x=self.conv1d(x.transpose(1,2)).transpose(1,2);x=F.relu(x)
        lstm_out,_=self.lstm(x);gru_out,_=self.gru(lstm_out);return self.output(gru_out[:,-1])

class AttentionLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim=128):
        super().__init__();self.lstm=nn.LSTM(input_dim,hidden_dim,batch_first=True)
        self.attention=nn.MultiheadAttention(hidden_dim,8,batch_first=True);self.output=nn.Linear(hidden_dim,1)
    def forward(self,x):
        lstm_out,_=self.lstm(x);attn_out,_=self.attention(lstm_out,lstm_out,lstm_out)
        return self.output(attn_out.mean(1))
