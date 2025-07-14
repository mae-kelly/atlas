import torch,torch.nn as nn,torch.nn.functional as F,numpy as np,math,time
from typing import Dict,List,Tuple,Optional

class HoltWintersCell(nn.Module):
    def __init__(self,embed_dim:int,num_seasons:int=12):
        super().__init__();self.embed_dim,self.num_seasons=embed_dim,num_seasons
        self.alpha,self.beta,self.gamma=nn.Parameter(torch.tensor(0.3)),nn.Parameter(torch.tensor(0.1)),nn.Parameter(torch.tensor(0.2))
        self.level_net,self.trend_net,self.seasonal_net=nn.Linear(embed_dim,embed_dim),nn.Linear(embed_dim,embed_dim),nn.Linear(embed_dim,embed_dim)
        self.register_buffer('seasonal_components',torch.randn(num_seasons,embed_dim))
    
    def forward(self,x:torch.Tensor,level_prev:torch.Tensor,trend_prev:torch.Tensor,season_idx:int)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        batch_size=x.shape[0];seasonal_curr=self.seasonal_components[season_idx%self.num_seasons].unsqueeze(0).expand(batch_size,-1)
        deseasonalized=x-seasonal_curr;level_new=torch.sigmoid(self.alpha)*deseasonalized+(1-torch.sigmoid(self.alpha))*(level_prev+trend_prev)
        level_new=self.level_net(level_new);trend_new=torch.sigmoid(self.beta)*(level_new-level_prev)+(1-torch.sigmoid(self.beta))*trend_prev
        trend_new=self.trend_net(trend_new);seasonal_update=torch.sigmoid(self.gamma)*(x-level_new)+(1-torch.sigmoid(self.gamma))*seasonal_curr
        seasonal_update=self.seasonal_net(seasonal_update)
        with torch.no_grad():self.seasonal_components[season_idx%self.num_seasons]=seasonal_update.mean(0)
        return level_new,trend_new,seasonal_update

class HelformerAttention(nn.Module):
    def __init__(self,embed_dim:int,num_heads:int=8,num_seasons:int=12):
        super().__init__();self.embed_dim,self.num_heads,self.head_dim,self.num_seasons=embed_dim,num_heads,embed_dim//num_heads,num_seasons
        self.q_proj,self.k_proj,self.v_proj,self.out_proj=nn.Linear(embed_dim,embed_dim),nn.Linear(embed_dim,embed_dim),nn.Linear(embed_dim,embed_dim),nn.Linear(embed_dim,embed_dim)
        self.hw_cell,self.temporal_encoding,self.scale=HoltWintersCell(embed_dim,num_seasons),nn.Embedding(num_seasons,embed_dim),1.0/math.sqrt(self.head_dim)
    
    def forward(self,x:torch.Tensor,mask:Optional[torch.Tensor]=None,level_state:Optional[torch.Tensor]=None,trend_state:Optional[torch.Tensor]=None,time_indices:Optional[torch.Tensor]=None)->Dict[str,torch.Tensor]:
        batch_size,seq_len,embed_dim=x.shape
        level_state=level_state or torch.zeros(batch_size,embed_dim,device=x.device)
        trend_state=trend_state or torch.zeros(batch_size,embed_dim,device=x.device)
        time_indices=time_indices or torch.arange(seq_len,device=x.device).unsqueeze(0).expand(batch_size,-1)
        hw_outputs,current_level,current_trend=[],level_state,trend_state
        for t in range(seq_len):
            season_idx=time_indices[:,t].mean().long().item()
            level_new,trend_new,seasonal=self.hw_cell(x[:,t,:],current_level,current_trend,season_idx)
            hw_outputs.append(level_new+trend_new+seasonal);current_level,current_trend=level_new,trend_new
        hw_features=torch.stack(hw_outputs,dim=1);temporal_emb=self.temporal_encoding(time_indices%self.num_seasons)
        enhanced_x=hw_features+temporal_emb
        q=self.q_proj(enhanced_x).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(enhanced_x).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(enhanced_x).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        scores=torch.matmul(q,k.transpose(-2,-1))*self.scale
        if mask is not None:scores=scores.masked_fill(mask.unsqueeze(1)==0,float('-inf'))
        attn_weights=F.softmax(scores,dim=-1);attn_output=torch.matmul(attn_weights,v)
        attn_output=attn_output.transpose(1,2).contiguous().view(batch_size,seq_len,embed_dim)
        return{'output':self.out_proj(attn_output),'attention_weights':attn_weights,'level_state':current_level,'trend_state':current_trend,'hw_features':hw_features}

class HelformerModel(nn.Module):
    def __init__(self,input_dim:int,embed_dim:int=512,num_heads:int=16,num_layers:int=6,num_classes:int=6,num_seasons:int=12,dropout:float=0.1,max_seq_len:int=1000):
        super().__init__();self.embed_dim,self.num_layers,self.num_seasons=embed_dim,num_layers,num_seasons
        self.input_embedding=nn.Linear(input_dim,embed_dim);self.pos_encoding=self._create_positional_encoding(max_seq_len,embed_dim)
        self.helformer_layers=nn.ModuleList([HelformerLayer(embed_dim,num_heads,dropout=dropout,num_seasons=num_seasons)for _ in range(num_layers)])
        self.classifier=nn.Sequential(nn.Linear(embed_dim,embed_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(embed_dim//2,num_classes))
        self.asset_type_head,self.confidence_head=nn.Linear(embed_dim,num_classes),nn.Sequential(nn.Linear(embed_dim,1),nn.Sigmoid())
        self.risk_return_head=nn.Sequential(nn.Linear(embed_dim,embed_dim//2),nn.GELU(),nn.Linear(embed_dim//2,1))
    
    def _create_positional_encoding(self,max_seq_len:int,embed_dim:int)->torch.Tensor:
        pe,position=torch.zeros(max_seq_len,embed_dim),torch.arange(0,max_seq_len).unsqueeze(1).float()
        div_term=torch.exp(torch.arange(0,embed_dim,2).float()*-(math.log(10000.0)/embed_dim))
        pe[:,0::2],pe[:,1::2]=torch.sin(position*div_term),torch.cos(position*div_term);return pe.unsqueeze(0)
    
    def forward(self,x:torch.Tensor,mask:Optional[torch.Tensor]=None,time_indices:Optional[torch.Tensor]=None)->Dict[str,torch.Tensor]:
        batch_size,seq_len,_=x.shape;x=self.input_embedding(x)
        if seq_len<=self.pos_encoding.shape[1]:x=x+self.pos_encoding[:,:seq_len,:].to(x.device)
        level_state,trend_state,attention_weights=None,None,[]
        for layer in self.helformer_layers:
            layer_output=layer(x,mask,level_state,trend_state,time_indices)
            x,level_state,trend_state=layer_output['output'],layer_output['level_state'],layer_output['trend_state']
            attention_weights.append(layer_output['attention_weights'])
        sequence_repr=x.mean(dim=1)if mask is None else(x*mask.any(dim=-1,keepdim=True).float()).sum(dim=1)/mask.any(dim=-1,keepdim=True).float().sum(dim=1)
        return{'predictions':self.classifier(sequence_repr),'asset_predictions':self.asset_type_head(sequence_repr),'confidence':self.confidence_head(sequence_repr),'risk_return_score':self.risk_return_head(sequence_repr),'sequence_representation':sequence_repr,'attention_weights':attention_weights,'level_state':level_state,'trend_state':trend_state}
    
    def predict_excess_return(self,x:torch.Tensor,current_portfolio_value:float=1.0)->Dict[str,float]:
        self.eval()
        with torch.no_grad():
            outputs=self.forward(x);risk_return_raw=outputs['risk_return_score'].squeeze()
            excess_return_multiplier=torch.tanh(risk_return_raw)*9.25;confidence_weight=outputs['confidence'].squeeze()
            predicted_excess_return=excess_return_multiplier*confidence_weight
            estimated_volatility=torch.std(predicted_excess_return)+0.01;sharpe_ratio=predicted_excess_return.mean()/estimated_volatility
            return{'excess_return_pct':float(predicted_excess_return.mean())*100,'expected_portfolio_value':current_portfolio_value*(1+float(predicted_excess_return.mean())),'estimated_sharpe_ratio':float(sharpe_ratio),'confidence':float(confidence_weight.mean()),'risk_score':float(torch.abs(risk_return_raw).mean())}

class HelformerLayer(nn.Module):
    def __init__(self,embed_dim:int,num_heads:int=8,ff_dim:int=None,dropout:float=0.1,num_seasons:int=12):
        super().__init__();ff_dim=ff_dim or 4*embed_dim;self.attention=HelformerAttention(embed_dim,num_heads,num_seasons)
        self.ff_network=nn.Sequential(nn.Linear(embed_dim,ff_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(ff_dim,embed_dim),nn.Dropout(dropout))
        self.norm1,self.norm2,self.dropout=nn.LayerNorm(embed_dim),nn.LayerNorm(embed_dim),nn.Dropout(dropout)
    
    def forward(self,x:torch.Tensor,mask:Optional[torch.Tensor]=None,level_state:Optional[torch.Tensor]=None,trend_state:Optional[torch.Tensor]=None,time_indices:Optional[torch.Tensor]=None)->Dict[str,torch.Tensor]:
        attn_output=self.attention(self.norm1(x),mask,level_state,trend_state,time_indices)
        x=x+self.dropout(attn_output['output']);ff_output=self.ff_network(self.norm2(x));x=x+ff_output
        return{'output':x,'attention_weights':attn_output['attention_weights'],'level_state':attn_output['level_state'],'trend_state':attn_output['trend_state']}

def create_helformer_for_trading(input_dim:int=27)->HelformerModel:
    return HelformerModel(input_dim=input_dim,embed_dim=512,num_heads=16,num_layers=6,num_classes=6,num_seasons=12,dropout=0.1,max_seq_len=1000)

def helformer_loss_function(outputs:Dict[str,torch.Tensor],targets:torch.Tensor,asset_targets:Optional[torch.Tensor]=None,return_targets:Optional[torch.Tensor]=None)->torch.Tensor:
    classification_loss=F.cross_entropy(outputs['predictions'],targets)
    asset_loss=F.cross_entropy(outputs['asset_predictions'],asset_targets)if asset_targets is not None else torch.tensor(0.0)
    return_loss=F.mse_loss(outputs['risk_return_score'].squeeze(),return_targets)if return_targets is not None else torch.tensor(0.0)
    confidence=outputs['confidence'].squeeze();correct_predictions=(outputs['predictions'].argmax(dim=1)==targets).float()
    confidence_loss=F.mse_loss(confidence,correct_predictions)
    return 1.0*classification_loss+0.8*asset_loss+2.0*return_loss+0.3*confidence_loss
