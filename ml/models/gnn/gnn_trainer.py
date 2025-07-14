import torch,torch.nn as nn,torch.optim as optim
from torch_geometric.loader import DataLoader
from .blockchain_analyzer import BlockchainTransactionAnalyzer
from loguru import logger

class GNNTrainer:
    def __init__(self,model,device='cuda'):
        self.model,self.device=model,torch.device(device if torch.cuda.is_available()else'cpu')
        self.model.to(self.device);self.optimizer=optim.Adam(self.model.parameters(),lr=0.001,weight_decay=5e-4)
        self.criterion=nn.CrossEntropyLoss()
    
    def train_epoch(self,data_loader):
        self.model.train();total_loss,correct,total=0,0,0
        for batch in data_loader:
            batch=batch.to(self.device);self.optimizer.zero_grad()
            if hasattr(self.model,'forward')and 'batch_graphs'in self.model.forward.__code__.co_varnames:
                out=self.model([batch])
            else:out=self.model(batch.x,batch.edge_index,batch.batch)
            loss=self.criterion(out,batch.y);loss.backward();self.optimizer.step()
            total_loss+=loss.item();pred=out.argmax(dim=1);correct+=(pred==batch.y).sum().item();total+=batch.y.size(0)
        return total_loss/len(data_loader),correct/total
    
    def evaluate(self,data_loader):
        self.model.eval();total_loss,correct,total=0,0,0
        with torch.no_grad():
            for batch in data_loader:
                batch=batch.to(self.device)
                if hasattr(self.model,'forward')and'batch_graphs'in self.model.forward.__code__.co_varnames:out=self.model([batch])
                else:out=self.model(batch.x,batch.edge_index,batch.batch)
                loss=self.criterion(out,batch.y);total_loss+=loss.item()
                pred=out.argmax(dim=1);correct+=(pred==batch.y).sum().item();total+=batch.y.size(0)
        return total_loss/len(data_loader),correct/total
    
    def save_model(self,path:str):
        torch.save({'model_state_dict':self.model.state_dict(),'optimizer_state_dict':self.optimizer.state_dict()},path)
        logger.info(f"💾 GNN model saved to {path}")
