import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from models import LSTMForecast
from mpi4py import MPI
import os, shutil, argparse, warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)
if shutil.which('latex'):
    print("Latex detected!",flush=True)
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"
else:
    print("Latex not detected!",flush=True)

from torch.nn.functional import mse_loss as MSELoss

from ServerOptimizers import FedAvg,FedAvgAdaptive,FedAdagrad,FedYogi,FedAdam
from ClientOptimizers import Prox, ProxAdam, Adam, AdamAMS
from utils import DatasetCleaner, ModelExtractor, set_seed

#args
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--state',type=str,default='CA')
parser.add_argument('--choice_local',type=int,choices = [0,1,2,3])
args = parser.parse_args()
 
# we use convention that the update should be 'added' to states
# do local gradient calcs accordingly
        
class CData:
    
    def __init__(self):
        
        self.n_features = 8
        self.seq_len = 12
        self.lookahead = 4
        self.batch_size = 16
        self.lr = 1e-3
        self.server_lr = 1e-2
        self.beta = 0.5
        self.beta_1 = 0.5
        self.beta_2 = 0.9
        self.beta_1s = 0.5
        self.beta_2s = 0.9
        self.eps = 1e-8
        self.n_clients = 12
        self.state = args.state
        self.train_test_split = 0.8
        self.local_epochs = 125
        self.global_epochs = 40
        self.net_hidden_size = 25
        self.n_lstm_layers = 1
        self.weight_decay = 1e-1
        self.test_every = 8
        self.save_at_end = True
        
def learn_model(comm,cData,local_kw,local_opt,local_name,global_kw,global_opt,global_name,model_kw,p_layers,p_name,device):
    
    # master function to do fed learning
    
    # set seed
    set_seed(10)
    
    rank = comm.Get_rank()
    total_clients = comm.Get_size()-1
    
    if rank == 0:
        # server
        model = LSTMForecast(**model_kw).to(device) if global_name!='FedAvgAdaptive' else [LSTMForecast(**model_kw).to(device) for _ in range(total_clients)]# server's aggregate model
        globalOpt = global_opt(model=model,n_clients=total_clients,**global_kw)
        for e_global in range(cData.global_epochs):
            for cidx in range(total_clients):
                if global_name == 'FedAvgAdaptive':
                    comm.Send([globalOpt.mes[cidx].get_flattened_params().astype(np.float64),MPI.DOUBLE],dest=cidx+1,tag=0)
                else:
                    comm.Send([globalOpt.me.get_flattened_params().astype(np.float64),MPI.DOUBLE],dest=cidx+1,tag=0)
                # comm.Send([globalOpt.c.astype(np.float64),MPI.DOUBLE],dest=cidx+1,tag=1)
            grads, deltaC = [], []
            for cidx in range(total_clients):
                buf, buf_C = np.empty(globalOpt.num_params,dtype=np.float64), np.empty(globalOpt.num_params,dtype=np.float64)
                comm.Recv([buf,MPI.DOUBLE],source=cidx+1,tag=0)
                # comm.Recv([buf_C,MPI.DOUBLE],source=cidx+1,tag=1)
                grads.append(buf.copy())
                # deltaC.append(buf_C.copy())
            globalOpt.aggregate_and_update(grads=grads)
            # globalOpt.c += np.mean(np.array(deltaC),axis=0)
        returnME = globalOpt.mes[0] if global_name=='FedAvgAdaptive' else globalOpt.me
        return None,returnME
    
    else:
        # clients
        client_id = rank - 1 # because 0 is server
        dset = DatasetCleaner(np.load(f"NREL{cData.state}dataset.npz")['data'],cidx=client_id,clientList=np.arange(comm.Get_size()-1).tolist(),seq_len=cData.seq_len,
                lookahead=cData.lookahead,train_test_split=cData.train_test_split,device=device)
        model = LSTMForecast(**model_kw).to(device) # client's local model
        localOpt = local_opt(model=model,p_layers=p_layers,**local_kw)
        mase_test = []
        for e_global in range(cData.global_epochs):
            buf = np.empty(localOpt.me.num_params,dtype=np.float64)
            comm.Recv([buf,MPI.DOUBLE],source=0,tag=0)
            if local_name in ['Adam','AdamAMS']:
                localOpt.me.set_flattened_params_shared(buf)
            # buf_c = np.empty(localOpt.me.num_params,dtype=np.float64)
            # comm.Recv([buf_c,MPI.DOUBLE],source=0,tag=1)
            before_update = localOpt.me.get_flattened_params()
            if p_name != 'All layers personalized':
                localOpt.reset_counter()
            for e_local in range(cData.local_epochs):
                input, label = dset.sample_train(cData.batch_size)
                localOpt.update(input,label,buf)
                # localOpt.update_variates()
            comm.Send([(localOpt.me.get_flattened_params()-before_update).astype(np.float64),MPI.DOUBLE],dest=0,tag=0)
            # comm.Send([localOpt.delta.astype(np.float64),MPI.DOUBLE],dest=0,tag=1)
            if (e_global+1) % cData.test_every == 0 or e_global == cData.global_epochs-1:
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
    localOptNames = [localOptNames[args.choice_local]]
    localOptKw = [localOptKw[args.choice_local]]
    
    # global optim partial config
    fedavg_kw = {'lr':cData.server_lr,'weights':None}
    fedavgadaptive_kw = {'lr':cData.server_lr,'beta':cData.beta,'eps':cData.eps,'q':5,'weights':None}
    fedadagrad_kw = {'lr':cData.server_lr,'beta_1':cData.beta_1s,'eps':cData.eps,'weights':None}
    fedyogi_kw = {'lr':cData.server_lr,'beta_1':cData.beta_1s,'beta_2':cData.beta_2s,'eps':cData.eps,'weights':None}
    fedadam_kw = {'lr':cData.server_lr,'beta_1':cData.beta_1s,'beta_2':cData.beta_2s,'eps':cData.eps,'weights':None}
    globalOptNames = [FedAvg,FedAvgAdaptive,FedAdagrad,FedYogi,FedAdam]
    globalOptKw = [fedavg_kw,fedavgadaptive_kw,fedadagrad_kw,fedyogi_kw,fedadam_kw]
    
    # personalization levels
    pers0 = [] # all shared
    pers1 = ['FCLayer1.weight','FCLayer1.bias','FCLayer2.weight','FCLayer3.weight','FCLayer2.bias','prelu1.weight','prelu2.weight'] # linear head personalized
    pers2 = [layerName for layerName,_ in dummyModel.named_parameters()] # all personalized
    # pLayers = [pers0,pers1,pers2]
    # pLayerNames = ['All layers shared','Linear head personalized','All layers personalized']
    pLayers = [pers1,pers2]
    pLayerNames = ['Linear head personalized','All layers personalized']
    
    # loop testing
    for pl,pn in zip(pLayers,pLayerNames):
        errMat = np.zeros(shape=(len(localOptNames),len(globalOptNames)))
        for li,(lo,lk) in enumerate(zip(localOptNames,localOptKw)):
            for gi,(go,gk) in enumerate(zip(globalOptNames,globalOptKw)):
                errors, me = learn_model(comm,cData,lk,lo,lo.__name__,gk,go,go.__name__,model_kw,pl,pn,device)
                if cData.save_at_end:
                    topdir = os.getcwd()+f"/experiments{cData.state}/{pn}_{go.__name__}_{lo.__name__}/{'server' if comm.Get_rank()==0 else f'client{comm.Get_rank()-1}'}"
                    expath = os.getcwd()+f"/experiments{cData.state}"
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
                        buf = np.empty(1,dtype=np.float32)
                        comm.Recv([buf,MPI.FLOAT],source=cidx+1)
                        cur_error += (1/(comm.Get_size()-1))*buf.item()
                errMat[li,gi] = cur_error 

        if comm.Get_rank() == 0:
            plt.imshow(errMat, origin='lower', cmap='viridis', alpha = 0.5, extent=[0, errMat.shape[1], 0, errMat.shape[0]])
            plt.xticks([0.5+i for i in range(len(globalOptNames))],[itm.__name__ for itm in globalOptNames],rotation=90)
            plt.yticks([0.5+i for i in range(len(localOptNames))],[itm.__name__ for itm in localOptNames])
            plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
            plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
            for i in range(errMat.shape[0]):
                for j in range(errMat.shape[1]):
                    plt.annotate(f"{errMat[i, j]:.4f}", xy=(0.5+j, 0.5+i), ha='center', va='center', color='black')
            states = {'NY':'New York','CA':'California','IL':'Illinois'}
            plt.title(r"Dataset:$\textbf{%s}$, Personalization:$\textbf{%s}$%sAverage test MASE across all clients"%(
                f'{states[cData.state]}',f'{pn}',f'\n'
            ))
            plt.colorbar()
            plt.grid(which='minor',color='k')
            plt.savefig(os.getcwd()+f'/experiments{cData.state}/errMat_{localOptNames[0].__name__}_{pn}.pdf',format='pdf',bbox_inches='tight') 
            plt.close()
            with open(expath+'/results.txt',"a") as file:
                file.write(f"\ndt={str(datetime.now())}\nFor global={[i.__name__ for i in globalOptNames]},\n local={[i.__name__ for i in localOptNames]},\n state={cData.state}, pers={pn} MASES are:\n")
                file.write(f"{errMat}\n")