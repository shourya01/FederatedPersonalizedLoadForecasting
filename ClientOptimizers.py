import torch
import torch.nn as nn
import numpy as np
from utils import ModelExtractor

class Prox:
    
    # proximal optimization for personalized optimization purposes
    
    def __init__(self,model=nn.Module,p_layers=[],lr=1e-3,weight_decay=1e-1):
        
        self.model = model
        self.optim = torch.optim.SGD(self.model.parameters(),lr=1e-3)
        self.me = ModelExtractor(self.model,p_layers)
        self.lr = lr
        self.weight_decay = weight_decay
        
        # number of params in model
        nparams = 0
        for _,v in self.model.named_parameters():
            nparams += torch.numel(v)
        self.nparams = nparams
        
        self.init_state()
        
    def init_state(self):
        
        self.x = self.me.get_flattened_params()
        
    def reset_counter(self):
        
        pass
        
    def update(self,inp,tar,target_params:np.ndarray,lossfn=nn.MSELoss(reduction='mean')):
        
        # populate .grad
        self.optim.zero_grad()
        lossfn(self.model(inp),tar).backward()
        
        # assuming here that the .grad params are populated
        grad_collector = []
        for _,m in self.model.named_parameters():
            grad_collector.append(m.grad.flatten())
        grad = torch.cat(grad_collector,dim=-1).detach().cpu().numpy()
        self.x -= self.lr*( grad + self.weight_decay*(self.x-target_params) )
        self.me.set_flattened_params_shared(self.x)
        
class ProxAdam:
    
    # replicate pytorch Adam for local training
    
    def __init__(self,model=nn.Module,p_layers=[],lr=1e-3,beta_1=0.9,beta_2=0.999,eps=1e-8,weight_decay=1e-8):
        
        self.model = model
        self.optim = torch.optim.SGD(self.model.parameters(),lr=1e-3)
        self.me = ModelExtractor(self.model,p_layers)
        self.update_count = 1
        
        # number of params in model
        nparams = 0
        for _,v in self.model.named_parameters():
            nparams += torch.numel(v)
        self.nparams = nparams
        
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.init_state()
        
    def init_state(self):
        
        self.x = self.me.get_flattened_params()
        self.m = np.zeros(self.nparams)
        self.v = np.ones(self.nparams)
        self.mhat = np.zeros(self.nparams)
        self.vhat = np.zeros(self.nparams)
        
    def reset_counter(self):
        
        self.update_count += 1
        
    def update(self,inp,tar,target_params:np.ndarray,lossfn=nn.MSELoss(reduction='mean')):
        
        # populate .grad
        self.optim.zero_grad()
        self.optim = torch.optim.SGD(self.model.parameters(),lr=1e-3)
        lossfn(self.model(inp),tar).backward()
        
        # assuming here that the .grad params are populated
        grad_collector = []
        for _,m in self.model.named_parameters():
            grad_collector.append(m.grad.flatten())
        grad = torch.cat(grad_collector,dim=-1).detach().cpu().numpy()+self.weight_decay*(self.x-target_params)
        self.m = self.beta_1*self.m + (1-self.beta_1)*grad
        self.v = self.beta_2*self.v + (1-self.beta_2)*np.square(grad)
        self.mhat = self.m / (1-np.power(self.beta_1,self.update_count))
        self.vhat = self.v / (1-np.power(self.beta_2,self.update_count))
        self.x -= self.lr*self.mhat / (np.sqrt(self.vhat) + self.eps)
        self.me.set_flattened_params_shared(self.x)
        self.update_count += 1

class Adam:
    
    # replicate pytorch Adam for local training
    
    def __init__(self,model=nn.Module,p_layers=[],lr=1e-3,beta_1=0.9,beta_2=0.999,eps=1e-8):
        
        self.model = model
        self.optim = torch.optim.SGD(self.model.parameters(),lr=1e-3)
        self.me = ModelExtractor(self.model,p_layers)
        self.update_count = 1
        
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        
        # number of params in model
        nparams = 0
        for _,v in self.model.named_parameters():
            nparams += torch.numel(v)
        self.nparams = nparams
        
        self.init_state()
        
    def init_state(self):
        
        self.x = self.me.get_flattened_params()
        self.m = np.zeros(self.nparams)
        self.v = np.ones(self.nparams)
        self.mhat = np.zeros(self.nparams)
        self.vhat = np.zeros(self.nparams)
        
    def reset_counter(self):
        
        self.update_count += 1
        
    def update(self,inp,tar,lossfn=nn.MSELoss(reduction='mean')):
        
        # populate .grad
        self.optim.zero_grad()
        lossfn(self.model(inp),tar).backward()
        
        # assuming here that the .grad params are populated
        grad_collector = []
        for _,m in self.model.named_parameters():
            grad_collector.append(m.grad.flatten())
        grad = torch.cat(grad_collector,dim=-1).detach().cpu().numpy()
        self.m = self.beta_1*self.m + (1-self.beta_1)*grad
        self.v = self.beta_2*self.v + (1-self.beta_2)*np.square(grad)
        self.mhat = self.m / (1-np.power(self.beta_1,self.update_count))
        self.vhat = self.v / (1-np.power(self.beta_2,self.update_count))
        self.x -= self.lr*self.mhat / (np.sqrt(self.vhat) + self.eps)
        self.me.set_flattened_params_shared(self.x)
        self.update_count += 1
        
        
class AdamAMS:
    
    # replicate pytorch AdamAMS for local training
    
    def __init__(self,model=nn.Module,p_layers=[],lr=1e-3,beta_1=0.9,beta_2=0.999,eps=1e-8):
        
        self.model = model
        self.optim = torch.optim.SGD(self.model.parameters(),lr=1e-3)
        self.me = ModelExtractor(self.model,p_layers)
        self.update_count = 1
        
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        
        # number of params in model
        nparams = 0
        for _,v in self.model.named_parameters():
            nparams += torch.numel(v)
        self.nparams = nparams
        
        self.init_state()
        
    def init_state(self):
        
        self.x = self.me.get_flattened_params()
        self.m = np.zeros(self.nparams)
        self.v = np.ones(self.nparams)
        self.mhat = np.zeros(self.nparams)
        self.vhat = np.zeros(self.nparams)
        self.vhatmax = (self.eps)*np.ones(self.nparams)
        
    def reset_counter(self):
        
        self.update_count += 1
        
    def update(self,inp,tar,lossfn=nn.MSELoss(reduction='mean')):
        
        # populate .grad
        self.optim.zero_grad()
        lossfn(self.model(inp),tar).backward()
        
        # assuming here that the .grad params are populated
        grad_collector = []
        for _,m in self.model.named_parameters():
            grad_collector.append(m.grad.flatten())
        grad = torch.cat(grad_collector,dim=-1).detach().cpu().numpy()
        self.m = self.beta_1*self.m + (1-self.beta_1)*grad
        self.v = self.beta_2*self.v + (1-self.beta_2)*np.square(grad)
        self.mhat = self.m / (1-np.power(self.beta_1,self.update_count))
        self.vhat = self.v / (1-np.power(self.beta_2,self.update_count))
        self.vhatmax = np.maximum(self.vhat,self.vhatmax)
        self.x -= self.lr*self.mhat / (np.sqrt(self.vhatmax) + self.eps)
        self.me.set_flattened_params_shared(self.x)
        self.update_count += 1