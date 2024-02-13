import torch
import torch.nn as nn
import numpy as np
from utils import ModelExtractor

class FedAvg:
    
    def __init__(self,model:nn.Module,n_clients=2,lr=1e-3,weights = None):
        self.me = ModelExtractor(model=model,pers_layers=[])
        self.n_clients = n_clients
        self.num_params = self.me.num_params
        self.lr = lr
        self.weights = weights
        self.model = model
        self.num_params = self.me.num_params
        self.init_states()
        
    def init_states(self):
        self.x = self.me.get_flattened_params()
        
    def aggregate_and_update(self,grads=None):
        if self.weights is None:
            grads = np.array(grads)
        else:
            grads = np.array([itm*self.weights[i] for i,itm in enumerate(grads)])
        self.x += self.lr*np.mean(grads,axis=0)
        self.me.set_flattened_params_all(self.x)
        
class FedAvgAdaptive:
    
    def __init__(self,model:list,n_clients=2,lr=1e-3,beta_1 = 0.9,beta_2=0.999,eps = 1e-8,q=5,weights = None):
        assert len(model) == n_clients, "Different num of model than clients!"
        self.mes = [ModelExtractor(model=modelItm,pers_layers=[]) for modelItm in model]
        self.n_clients = n_clients
        self.num_params = self.mes[0].num_params
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weights = weights
        self.q = q # information sharing interval
        self.update_count = 1 # internal counter for how many updates are called
        self.num_params = self.mes[0].num_params
        self.init_states()
        
    def init_states(self):
        self.x = [me.get_flattened_params() for me in self.mes]
        self.m = [np.zeros((self.num_params,)) for _ in range(self.n_clients)]
        self.client_vars = [np.ones((self.num_params,)) for _ in range(self.n_clients)]
        
    def aggregate_and_update(self, grads=None):
        for idx in range(len(self.client_vars)):
            self.m[idx] = self.beta_1*self.m[idx] + (1-self.beta_1)*grads[idx]
            self.client_vars[idx] = self.beta_2*self.client_vars[idx] + (1-self.beta_2)*np.square(grads[idx])
        if self.update_count % self.q != 0:
            for idx in range(len(self.client_vars)):
                self.x[idx] += self.lr*self.m[idx]/(np.sqrt(self.client_vars[idx])+self.eps)
                self.mes[idx].set_flattened_params_all(self.x[idx])
        else:
            if self.weights is None:
                grads = np.array(grads)
            else:
                grads = np.array([itm*self.weights[i] for i,itm in enumerate(grads)])
            for idx in range(len(self.client_vars)):
                self.x[idx] += np.mean(grads,axis=0) + self.lr*self.m[idx]/(np.sqrt(self.client_vars[idx])+self.eps)
                self.mes[idx].set_flattened_params_all(self.x[idx])
        self.update_count += 1
        
        
class FedAdagrad:
    
    def __init__(self,model:nn.Module,n_clients=2,lr=1e-3,beta_1 = 0.9, eps = 1e-8,weights = None):
        self.me = ModelExtractor(model=model,pers_layers=[])
        self.n_clients = n_clients
        self.num_params = self.me.num_params
        self.lr = lr
        self.beta_1 = beta_1
        self.eps = eps
        self.weights = weights
        self.num_params = self.me.num_params
        self.init_states()
        
    def init_states(self):
        self.x = self.me.get_flattened_params()
        self.m = np.zeros(self.num_params)
        self.v = np.ones(self.num_params)
        
    def aggregate_and_update(self, grads=None):
        if self.weights is None:
            grads = np.array(grads)
        else:
            grads = np.array([itm*self.weights[i] for i,itm in enumerate(grads)])
        self.m = self.beta_1*self.m + (1-self.beta_1)*self.lr*np.mean(grads,axis=0)
        self.v += np.square(np.mean(grads,axis=0))
        self.x += self.lr*self.m/(np.sqrt(self.v)+self.eps)
        self.me.set_flattened_params_all(self.x)
        
class FedYogi:
    
    def __init__(self,model:nn.Module,n_clients=2,lr=1e-3,beta_1 = 0.9,beta_2 = 0.999, eps = 1e-8,weights = None):
        self.me = ModelExtractor(model=model,pers_layers=[])
        self.n_clients = n_clients
        self.num_params = self.me.num_params
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weights = weights
        self.num_params = self.me.num_params
        self.init_states()
        
    def init_states(self):
        self.x = self.me.get_flattened_params()
        self.m = np.zeros(self.num_params)
        self.v = np.ones(self.num_params)
        
    def aggregate_and_update(self, grads=None):
        if self.weights is None:
            grads = np.array(grads)
        else:
            grads = np.array([itm*self.weights[i] for i,itm in enumerate(grads)])
        meanGrad = np.mean(grads,axis=0)
        sqGrad = np.square(meanGrad)
        self.m = self.beta_1*self.m + (1-self.beta_1)*self.lr*meanGrad
        self.v = self.beta_2*self.v - (1-self.beta_2)*self.lr*sqGrad*np.sign(self.v-sqGrad)
        self.x += self.lr*self.m/(np.sqrt(self.v)+self.eps)
        self.me.set_flattened_params_all(self.x)
        
class FedAdam:
    
    def __init__(self,model:nn.Module,n_clients=2,lr=1e-3,beta_1 = 0.9,beta_2 = 0.999, eps = 1e-8,weights = None):
        self.me = ModelExtractor(model=model,pers_layers=[])
        self.n_clients = n_clients
        self.num_params = self.me.num_params
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weights = weights
        self.num_params = self.me.num_params
        self.init_states()
        
    def init_states(self):
        self.x = self.me.get_flattened_params()
        self.m = np.zeros(self.num_params)
        self.v = np.ones(self.num_params)
        
    def aggregate_and_update(self, grads=None):
        if self.weights is None:
            grads = np.array(grads)
        else:
            grads = np.array([itm*self.weights[i] for i,itm in enumerate(grads)])
        self.m = self.beta_1*self.m + (1-self.beta_1)*self.lr*np.mean(grads,axis=0)
        self.v = self.beta_2*self.v + (1-self.beta_2)*self.lr*np.square(np.mean(grads,axis=0))
        self.x += self.lr*self.m/(np.sqrt(self.v)+self.eps)
        self.me.set_flattened_params_all(self.x)