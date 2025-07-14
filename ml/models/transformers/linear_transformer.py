import torch,torch.nn as nn,torch.nn.functional as F,numpy as np,math
from typing import Optional
class LinearAttention(nn.Module):
    def __init__(self,dim,heads=8):
        super().__init__();self.heads,self.dim,self.head_dim=heads,dim,dim//heads
        self.q,self.k,self.v,self.proj=nn.Linear(dim,dim),nn.Linear(dim,dim),nn.Linear(dim,dim),nn.Linear(dim,dim)
    def forward(self,x):
        b,n,d=x.shape;q,k,v=self.q(x),self.k(x),self.v(x)
        q,k,v=map(lambda t:t.view(b,n,self.heads,self.head_dim).transpose(1,2),[q,k,v])
        q,k=F.elu(q)+1,F.elu(k)+1;kv=torch.einsum('bhnd,bhnf->bhdf',k,v);out=torch.einsum('bhnd,bhdf->bhnf',q,kv)
        return self.proj(out.transpose(1,2).reshape(b,n,d))
class TemporalFusionTransformer(nn.Module):
    def __init__(self,input_dim,hidden_dim=256,heads=8,layers=6):
        super().__init__();self.emb=nn.Linear(input_dim,hidden_dim)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(hidden_dim,heads,hidden_dim*4,0.1,batch_first=True)for _ in range(layers)])
        self.gating=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GLU(dim=-1))
        self.out=nn.Linear(hidden_dim//2,1)
    def forward(self,x):
        x=self.emb(x);[x:=layer(x)for layer in self.layers];x=self.gating(x);return self.out(x.mean(1))
class AutoformerLayer(nn.Module):
    def __init__(self,dim,seq_len,factor=1):
        super().__init__();self.factor,self.seq_len=factor,seq_len
        self.decomp=nn.AvgPool1d(kernel_size=3,stride=1,padding=1)
        self.attn=LinearAttention(dim);self.norm1,self.norm2=nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self,x):
        trend=self.decomp(x.transpose(1,2)).transpose(1,2);seasonal=x-trend
        seasonal=seasonal+self.attn(self.norm1(seasonal));seasonal=seasonal+self.ff(self.norm2(seasonal))
        return seasonal+trend
class FEDformer(nn.Module):
    def __init__(self,seq_len,dim,modes=32):
        super().__init__();self.modes,self.seq_len=modes,seq_len
        self.fourier_layer=nn.Parameter(torch.randn(dim,modes,dtype=torch.cfloat))
        self.conv=nn.Conv1d(dim,dim,1);self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        b,n,d=x.shape;x_ft=torch.fft.rfft(x,dim=1);x_ft[:,:self.modes,:]*=self.fourier_layer
        x=torch.fft.irfft(x_ft,n=n,dim=1);return self.norm(x+self.conv(x.transpose(1,2)).transpose(1,2))
class Informer(nn.Module):
    def __init__(self,dim,heads,factor=5):
        super().__init__();self.factor,self.heads=factor,heads
        self.q,self.k,self.v=nn.Linear(dim,dim),nn.Linear(dim,dim),nn.Linear(dim,dim)
        self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        b,n,d=x.shape;q,k,v=self.q(x),self.k(x),self.v(x)
        u=self.factor*np.ceil(np.log(n)).astype('int').item()
        scores=torch.einsum('bnd,bmd->bnm',q,k)/math.sqrt(d//self.heads)
        M=scores.max(dim=-1)[0]-scores.mean(dim=-1);M_top=M.topk(u,sorted=False)[1]
        q_reduced=q[torch.arange(b)[:,None,None],M_top[:,None,:],:]
        context=torch.einsum('bund,bmd->bunm',q_reduced,k).softmax(-1)@v
        return self.proj(context.mean(2))