import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from models import LSTMForecast
from mpi4py import MPI
import os
import argparse

from torch.nn.functional import mse_loss as MSELoss

from ServerOptimizers import FedAvg,FedAvgAdaptive,FedAdagrad,FedYogi,FedAdam
from ClientOptimizers import Prox, ProxAdam, Adam, AdamAMS
from utils import DatasetCleaner, ModelExtractor, set_seed

#args
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--state',type=str,default='CA')
args = parser.parse_args()
 
# we use convention that the update should be 'added' to states
# do local gradient calcs accordingly
        
class CData:
    
    def __init__(self):
        
        self.n_features = 8
        self.seq_len = 12
        self.lookahead = 4
        self.batch_size = 64
        self.lr = 5e-4
        self.server_lr = 1e-3
        self.beta = 0.9
        self.beta_1 = 0.9
        self.beta_2 = 0.9
        self.eps = 1e-8
        self.n_clients = 12
        self.state = args.state
        self.train_test_split = 0.8
        self.local_epochs = 100
        self.global_epochs = 2
        self.net_hidden_size = 25
        self.n_lstm_layers = 2
        self.weight_decay = 1e-1
        self.test_every = 1
        self.save_at_end = True
        
def learn_model(comm,cData,local_kw,local_opt,local_name,global_kw,global_opt,global_name,model_kw,p_layers,p_name,device):
    
    # set seed
    set_seed(10)
    
    rank = comm.Get_rank()
    total_clients = comm.Get_size()-1
    
    if rank == 0:
        # server
        if global_name == 'FedAvgAdaptive':
            model = [LSTMForecast(**model_kw).to(device) for _ in range(total_clients)] # server's aggregate model
        else:
            model = LSTMForecast(**model_kw).to(device) # server's aggregate model
        globalOpt = global_opt(model=model,n_clients=total_clients,**global_kw)
        for e_global in range(cData.global_epochs):
            for cidx in range(total_clients):
                if global_name == 'FedAvgAdaptive':
                    comm.Send([globalOpt.mes[cidx].get_flattened_params().astype(np.float64),MPI.DOUBLE],dest=cidx+1,tag=0)
                else:
                    comm.Send([globalOpt.me.get_flattened_params().astype(np.float64),MPI.DOUBLE],dest=cidx+1,tag=0)
            grads = []
            for cidx in range(total_clients):
                buf = np.empty(globalOpt.num_params,dtype=np.float64)
                comm.Recv([buf,MPI.DOUBLE],source=cidx+1,tag=0)
                grads.append(buf.copy())
            globalOpt.aggregate_and_update(grads=grads)
        returnME = globalOpt.mes[0] if global_name=='FedAvgAdaptive' else globalOpt.me
        return None,returnME
    
    else:
        # clients
        client_id = rank - 1 # because 0 is server
        dset = DatasetCleaner(np.load(f"NREL{cData.state}dataset.npz")['data'],cidx=client_id,clientList=[3*i for i in range(comm.Get_size()-1)],seq_len=cData.seq_len,
                lookahead=cData.lookahead,train_test_split=cData.train_test_split,device=device)
        model = LSTMForecast(**model_kw).to(device) # client's local model
        localOpt = local_opt(model=model,p_layers=p_layers,**local_kw)
        mase_test = []
        for e_global in range(cData.global_epochs):
            buf = np.empty(localOpt.me.num_params,dtype=np.float64)
            comm.Recv([buf,MPI.DOUBLE],source=0,tag=0)
            if local_name in ['Adam','AdamAMS']:
                localOpt.me.set_flattened_params_shared(buf)
            before_update = localOpt.me.get_flattened_params()
            for e_local in range(cData.local_epochs):
                input, label = dset.sample_train(cData.batch_size)
                loss = MSELoss(model(input), label)
                localOpt.me.unset_param_grad() # zero out gradient fields
                torch.mean(loss).backward() # do a backward call
                if local_name in ['Prox','ProxAdam']:
                    localOpt.update(buf)
                else:
                    localOpt.update()
            comm.Send([(localOpt.me.get_flattened_params()-before_update).astype(np.float64),MPI.DOUBLE],dest=0,tag=0)
            if e_global % cData.test_every == 0:
                mase = lambda y,x,p: np.mean(np.abs(x - y)) / np.mean(np.abs(x - p))
                test_in, test_out, test_persistence = dset.get_test_dset()
                loss_val = mase(model(test_in).detach().cpu().flatten().numpy()*(dset.pmax-dset.pmin)+dset.pmin,test_out.cpu().flatten().numpy(),test_persistence.cpu().flatten().numpy()).item()
                print(f"Pers:{p_name}, local:{local_name}, global: {global_name}, epoch: {e_global+1}, client: {client_id+1}, test MASE loss is {loss_val}.",flush=True)
                mase_test.append(loss_val)
        return mase_test,localOpt.me

        
if __name__=="__main__":
    
    # parallelization stuff with MPI
    comm = MPI.COMM_WORLD
    assert comm.Get_size() > 1, 'need at least 2 MPI processes'
    cData = CData()
    
    print(f"Comm_rank is {comm.Get_rank()} and comm_size is {comm.Get_size()}.",flush=True)
    
    # set device
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        dCount = torch.cuda.device_count()
        dGroup = np.array_split(np.arange(comm.Get_size()),dCount)
        for idx,itm in enumerate(dGroup):
            if comm.Get_rank() in itm:
                device = f'cuda:{idx}'
    
    # model config
    model_kw = {'n_features':cData.n_features,'n_lookback':cData.seq_len,'n_lstm_layers':cData.n_lstm_layers,'n_hidden_size':cData.net_hidden_size}
    dummyModel = LSTMForecast(**model_kw) # for extracting layer data
    
    # local optim partial config
    prox_kw = {'lr':cData.lr,'weight_decay':cData.weight_decay}
    proxadam_kw = {'lr':cData.lr,'beta_1':cData.beta_1,'beta_2':cData.beta_2,'eps':cData.eps,'weight_decay':cData.weight_decay}
    adam_kw = {'lr':cData.lr,'beta_1':cData.beta_1,'beta_2':cData.beta_2,'eps':cData.eps}
    adamams_kw = {'lr':cData.lr,'beta_1':cData.beta_1,'beta_2':cData.beta_2,'eps':cData.eps}
    localOptNames = [Prox,ProxAdam,Adam,AdamAMS]
    localOptKw = [prox_kw,proxadam_kw,adam_kw,adamams_kw]
    
    # global optim partial config
    fedavg_kw = {'lr':cData.server_lr,'weights':None}
    fedavgadaptive_kw = {'lr':cData.server_lr,'beta':cData.beta,'eps':cData.eps,'q':5,'weights':None}
    fedadagrad_kw = {'lr':cData.server_lr,'beta_1':cData.beta_1,'eps':cData.eps,'weights':None}
    fedyogi_kw = {'lr':cData.server_lr,'beta_1':cData.beta_1,'beta_2':cData.beta_2,'eps':cData.eps,'weights':None}
    fedadam_kw = {'lr':cData.server_lr,'beta_1':cData.beta_1,'beta_2':cData.beta_2,'eps':cData.eps,'weights':None}
    globalOptNames = [FedAvg,FedAvgAdaptive,FedAdagrad,FedYogi,FedAdam]
    globalOptKw = [fedavg_kw,fedavgadaptive_kw,fedadagrad_kw,fedyogi_kw,fedadam_kw]
    
    # personalization levels
    pers0 = [] # all shared
    pers1 = ['FCLayer1.weight','FCLayer1.bias','FCLayer2.weight','FCLayer3.weight','FCLayer2.bias','prelu1.weight','prelu2.weight'] # linear head personalized
    pers2 = [layerName for layerName,_ in dummyModel.named_parameters()] # all personalized
    pLayers = [pers0,pers1,pers2]
    pLayerNames = ['allShared','linearHeadPersonalized','allPersonalized']
    
    # loop testing
    for pl,pn in zip(pLayers,pLayerNames):
        errMat = np.zeros(shape=(len(localOptNames),len(globalOptNames)))
        for gi,(go,gk) in enumerate(zip(globalOptNames,globalOptKw)):
            for li,(lo,lk) in enumerate(zip(localOptNames,localOptKw)):
                errors, me = learn_model(comm,cData,lk,lo,lo.__name__,gk,go,go.__name__,model_kw,pl,pn,device)
                if cData.save_at_end:
                    topdir = os.getcwd()+f"/experiments{cData.state}/{pn}_{go.__name__}_{lo.__name__}/{'server' if comm.Get_rank()==0 else f'client{comm.Get_rank()-1}'}"
                    if not os.path.exists(topdir):
                        os.makedirs(topdir)
                    torch.save(me.model.state_dict(),topdir+'/model.pth')
                cur_error = np.array(errors[-1],dtype=np.float32) if errors is not None else np.array(0,dtype=np.float32)
                if comm.Get_rank() > 0:
                    # send the error
                    comm.Send([cur_error,MPI.FLOAT],dest=0)
                    if cData.save_at_end:
                        plt.plot(errors)
                        plt.xlabel(f'Global epochs x{cData.test_every}')
                        plt.title('Local test MASE loss')
                        plt.savefig(topdir+'/mase.pdf',format='pdf',bbox_inches='tight')
                        plt.close()
                else:
                    for cidx in range(comm.Get_size()-1):
                        buf = np.array(0.).astype(np.float32)
                        comm.Recv([buf,MPI.FLOAT],source=cidx+1)
                        cur_error += (1/(comm.Get_size()-1))*buf
                errMat[li,gi] = cur_error
                comm.Barrier()
                
                # # send err matrix
                # global_sum = np.zeros_like(errMat,dtype=np.float64)
                # comm.Reduce([errMat,MPI.DOUBLE],[global_sum,MPI.DOUBLE],op=MPI.SUM,root=0)
        # plot mase matrix
        if comm.Get_rank() == 0:
            # recv data
            # errMat = None
            # global_sum = np.zeros_like(errMat,dtype=np.float64)
            # comm.Reduce([errMat,MPI.DOUBLE],[global_sum,MPI.DOUBLE],op=MPI.SUM,root=0)
            # errMat = global_sum
            plt.imshow(errMat, origin='lower', cmap='viridis', alpha = 0.5, extent=[0, errMat.shape[1], 0, errMat.shape[0]])
            plt.xticks([0.5+i for i in range(len(globalOptNames))],[itm.__name__ for itm in globalOptNames],rotation=90)
            plt.yticks([0.5+i for i in range(len(localOptNames))],[itm.__name__ for itm in localOptNames])
            for i in range(errMat.shape[0]):
                for j in range(errMat.shape[1]):
                    plt.annotate(f"{errMat[i, j]:.2f}", xy=(0.5+j, 0.5+i), ha='center', va='center', color='black')
            plt.title(r"Pers:$\mathbf{%s}$, DSET:$\mathbf{%s}$ Avg. final MASE Errors%slocalEpochs:%s, globalEpochs:%s%sModel:LSTM, lookback:%s, lookahead:%s%sclients:%s, batch:%s"%(
                f'{pn}',f'{cData.state}',f'\n',f'{cData.local_epochs}',f'{cData.global_epochs}',f'\n',f'{cData.seq_len}',f'{cData.lookahead}',f'\n',f'{comm.Get_size()-1}',f'{cData.batch_size}'
            ))
            plt.colorbar()
            plt.savefig(os.getcwd()+f'/experiments{cData.state}/errMat{pn}.pdf',format='pdf',bbox_inches='tight') 
            plt.close()