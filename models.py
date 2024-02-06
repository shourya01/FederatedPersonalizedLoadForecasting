import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    
    def __init__(self,n_features,n_lookback,n_lstm_layers,n_hidden_size,lookahead):
        super(LSTMForecast,self).__init__()
        self.n_features = n_features
        self.lookback = n_lookback
        self.n_lstm_layers = n_lstm_layers
        self.n_hidden_size = n_hidden_size
        self.lookahead = lookahead
        self.fcnn_in_size = self.n_hidden_size*(self.lookback+self.lookahead)
        
        # LSTM
        self.lstm_model = nn.LSTM(input_size=n_features,hidden_size=n_hidden_size,num_layers=n_lstm_layers,batch_first=True,bidirectional=False)
        
        # FCNN 
        self.FCLayer1 = nn.Linear(self.fcnn_in_size,self.fcnn_in_size//2)
        self.FCLayer2 = nn.Linear(self.fcnn_in_size//2,1)
        self.prelu1 = nn.PReLU(self.fcnn_in_size//2)
        
        
    def forward(self,x):
        
        # allow only for batched inputs
        if self.n_features > 1:
            if len(x.shape) != 3:
                raise ValueError('Only accepting batched input!')
            
        # extend x along specifc dimension with zeros
        x = torch.cat([x,torch.zeros(x.size(0),self.lookahead,x.size(2)).to(x.device)],dim=1)
            
        # initialize output and cell state
        ht,ct = self._init_hidden(x.shape,x.device)
        
        # get output of LSTM
        x, (ht,ct) = self.lstm_model(x,(ht.detach(),ct.detach()))
        
        # pass through fully connected layers
        return self._forward_fcnn(torch.cat(x.unbind(dim=1),dim=1))
    
    def _init_hidden(self,shape,device):
        
        # define initial h0 and c0 as zeros
        
        ht = torch.zeros((self.n_lstm_layers,shape[0],self.n_hidden_size)).to(device)
        ct = torch.zeros((self.n_lstm_layers,shape[0],self.n_hidden_size)).to(device)
        
        return ht,ct
        
    def _forward_fcnn(self,x):
        
        # pass LSTM outputs through a FCNN
        x = self.FCLayer1(x)
        x = self.prelu1(x)
        x = self.FCLayer2(x)
        
        # rake absolute value of output to ensure non-negativity
        return x.abs()