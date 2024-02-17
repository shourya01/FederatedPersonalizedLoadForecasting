import torch
import torch.nn as nn
import numpy as np
import random

class ModelExtractor:
    
    # accepts torch model and is capable of getting and setting
    # parameters in flattened numpy array format 
    # Can also selectively update non-personalized layers
    # You need to give it the names of personalized layers
    # as a list
    
    def __init__(self,model:nn.Module,pers_layers:list=[]):
        
        self.model = model
        self.pers_layers = pers_layers
        self.num_params = self.get_flattened_params().size
        self.param_names = [k for k,_ in self.model.named_parameters()]
        if pers_layers is None or pers_layers == []:
            self.pers_layers = []
    
    def get_flattened_params(self):
        
        return nn.utils.parameters_to_vector(self.model.parameters()).detach().cpu().numpy()
    
    def set_flattened_params_all(self,fparams):
        
        pointer = 0
        fp = torch.tensor(fparams).to(next(self.model.parameters()).dtype).to(next(self.model.parameters()).device)
        for _,v in self.model.named_parameters():
            v.data.copy_(fp[pointer:pointer+v.numel()].view_as(v))
            pointer += v.numel()
        
    def set_flattened_params_shared(self,fparams):
        
        pointer = 0
        fp = torch.tensor(fparams).to(next(self.model.parameters()).dtype).to(next(self.model.parameters()).device)
        for k,v in self.model.named_parameters():
            if k not in self.pers_layers:
                v.data.copy_(fp[pointer:pointer+v.numel()].view_as(v))
            pointer += v.numel()
    
    def gen_mask_for_slayers(self):
        
        # generate a mask for personalized layers
        maskVals = []
        for k,v in self.model.named_parameters():
            mask = np.zeros(torch.numel(v)) if k in self.pers_layers else np.ones(torch.numel(v))
            maskVals.append(mask)
        return np.concatenate(maskVals,axis=0)
        
class DatasetCleaner:
    
    # ingest numpy file, output random samples of train, or test set
    
    def __init__(self,dset,cidx=0,clientList=[],seq_len=12,lookahead=4,train_test_split=0.8,dtype=torch.float32,device='cpu'):
        
        self.dset = dset[:,:,:].copy()
        self.cidx = cidx
        if (clientList is None) or (clientList==[]):
            self.clientList = np.arange(self.dset.shape[0])
        else:
            self.clientList = clientList
        self.seq_len = seq_len
        self.lookahead = lookahead
        self.tts = train_test_split
        self.device = device
        self.dtype = dtype
        
        self.train_test_split()
        self.gen_dset(self.cidx)
        
    def train_test_split(self):
        
        self.dset_train = self.dset[:,:int(self.tts*self.dset.shape[1]),:].copy()
        self.dset_test = self.dset[:,int(self.tts*self.dset.shape[1]):,:].copy()
        self.dset_test_unscaled = self.dset_test.copy()
        minval,maxval = [], []
        
        for fidx in range(self.dset.shape[-1]):
            minval.append(self.dset_train[self.clientList,:,fidx].min())
            maxval.append(self.dset_train[self.clientList,:,fidx].max())
            
            if minval[-1] == maxval[-1]:
                self.dset_train[:,:,fidx], self.dset_test[:,:,fidx] = 1,1
            else:
                self.dset_train[:,:,fidx] = (self.dset_train[:,:,fidx] - minval[-1]) / (maxval[-1]-minval[-1])
                self.dset_test[:,:,fidx] = (self.dset_test[:,:,fidx] - minval[-1]) / (maxval[-1]-minval[-1])
                
        self.minval, self.maxval = minval, maxval
        self.pmin, self.pmax = minval[0], maxval[0]
                
    def gen_dset(self,cidx):
        
        inp, out = [], []
        cidxNew = self.clientList[cidx]
        for tidx in range(self.dset_train[cidxNew,:,:].shape[0]-self.seq_len-self.lookahead):
            inp.append(self.dset_train[cidxNew,tidx:tidx+self.seq_len,:])
            out.append(self.dset_train[cidxNew,tidx+self.seq_len+self.lookahead,[0]])
        self.train_in, self.train_out = torch.tensor(np.array(inp)).to(self.dtype).to(self.device), torch.tensor(np.array(out)).to(self.dtype).to(self.device)
        
        inp, out = [], []
        persistence = []
        for tidx in range(self.dset_test[cidxNew,:,:].shape[0]-self.seq_len-self.lookahead):
            inp.append(self.dset_test[cidxNew,tidx:tidx+self.seq_len,:])
            out.append(self.dset_test_unscaled[cidxNew,tidx+self.seq_len+self.lookahead,[0]])
            persistence.append(self.dset_test_unscaled[cidxNew,tidx+self.seq_len-1,[0]])
        self.test_in, self.test_out = torch.tensor(np.array(inp)).to(self.dtype).to(self.device), torch.tensor(np.array(out)).to(self.dtype).to(self.device)
        self.test_persistence = torch.tensor(np.array(persistence)).to(self.dtype).to(self.device)
        
    def sample_train(self,BS):
        
        batch_idx = np.random.choice(self.train_in.shape[0],BS,replace=False)
        return self.train_in[batch_idx], self.train_out[batch_idx]
    
    def unscale(self, x):
        
        return x*(self.maxval[0]-self.minval[0])+self.minval[0]
    
    def get_test_dset(self):
        
        return self.test_in, self.test_out, self.test_persistence
    
def set_seed(seed=233):
    
    # for reproducability and ensuring all algos are fed
    # the same minibatches for consistency
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False