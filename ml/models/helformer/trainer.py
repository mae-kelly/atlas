import torch,torch.nn as nn,numpy as np,time
from .helformer_core import create_helformer_for_trading,helformer_loss_function

class HelformerTrainer:
    def __init__(self,model_config:dict):
        self.model=create_helformer_for_trading(input_dim=model_config.get('input_dim',27))
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=3e-4,weight_decay=0.01)
        self.scheduler=torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=1e-3,epochs=100,steps_per_epoch=100)
        self.device=torch.device('cuda'if torch.cuda.is_available()else'cpu');self.model.to(self.device)
    
    def train_epoch(self,data_loader):
        self.model.train();total_loss,num_batches=0,0
        for batch_x,batch_y in data_loader:
            batch_x,batch_y=batch_x.to(self.device),batch_y.to(self.device);self.optimizer.zero_grad()
            outputs=self.model(batch_x);loss=helformer_loss_function(outputs,batch_y);loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);self.optimizer.step();self.scheduler.step()
            total_loss+=loss.item();num_batches+=1
        return total_loss/num_batches
    
    def predict_returns(self,market_data:torch.Tensor)->dict:
        self.model.eval()
        with torch.no_grad():return self.model.predict_excess_return(market_data.to(self.device))
    
    def save_model(self,path:str):torch.save({'model_state_dict':self.model.state_dict(),'optimizer_state_dict':self.optimizer.state_dict()},path)
    def load_model(self,path:str):checkpoint=torch.load(path);self.model.load_state_dict(checkpoint['model_state_dict']);self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
