import pandas as pd
from inspect import signature
import numpy as np
import glob
import itertools
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import seaborn as sns
import matplotlib.pyplot as plt
from MarkovModel import*
from  Full_model_decisionfocussed import *
from VRP_LSTMmodel import *
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(filename='PredictOptimize.log', 
level=logging.INFO,format=formatter)

npzfile = np.load("daily_stops.npz",allow_pickle=True,)
stops = npzfile['stops_list'] # 201 length list indicating which stops are active for each day
n_vehicles= npzfile['nr_vehicles'] # n_vehicles for each day
weekday= npzfile['weekday'] # categorical input
capacities = npzfile['capacities_list']# vehicle capacity
demands = npzfile['demands_list'] # demands of each active stops
npzfile = np.load("daily_routematrix.npz", allow_pickle=True)
opmat = npzfile['incidence_matrices'] # solutions for each day as an incidence matrix
stop_wise_days = npzfile['stop_wise_active'] 
distance_mat = np.load("Distancematrix.npy")
edge_mat = np.load("edge_category.npy")


test_days = [
        154,160, 166, 173, 180, 187, 194,
         155,161, 167, 174, 181, 188, 195,
         149,156, 168, 175, 182, 189, 196,
         150,162, 169, 176, 183, 190, 197,
         157,163, 170, 177, 184, 191, 198, 
         158, 164, 171, 178, 185, 192, 199,
         159, 165, 172, 179, 186, 193, 200]


net=  MarkovwthStopembedding  #OnlyMarkov #MarkovwthStopembedding 
lr,epochs,lookback = (0.01,15,127)
mu = 10 # 1e-2

for loss_type in ["Relu","Squared"]:
        model = PredictOptimzeVRP(epochs= epochs,lookback_period=lookback,mu= mu,
        loss_type= loss_type,
        stop_embedding= True,stop_embedding_size = 40,n_features=1,
        lr=lr,net=  net,capacitated= False,relaxed =False)
        T = 149
        model.fit(opmat[:T],weekday[:T], stops[:T],n_vehicles[:T], distance_mat, 
        stop_wise_days, demands[:T], capacities[:T])
        for t in test_days:
                lst = []
                ev = model.evaluation(distance_mat,stops[[t]],weekday[[t]],
                n_vehicles[[t]],opmat[:(t+1)],stop_wise_days,demands[t], capacities[t:(t+1)],
                capacitated=True)

                lst.append({"Day":t,
                        "lookback":lookback,
                        "epochs":epochs,"lr": lr,
                        "mu":mu,
                        "Model":str(net),"loss":loss_type,
                        "bceloss":ev[2],
                        "Arc Difference":ev[0][0],"Arc Difference(%)":ev[0][1],
                        "Route Difference":ev[1][0],
                        "Route Difference(%)":ev[1][1],
                        "Training AD":ev[3],
                        "Distance":ev[4],
                        "Comment":ev[5] })
                df = pd.DataFrame(lst)
                filename = "DecisionFocusedModels.csv"

                with open(filename, 'a') as f:
                        df.to_csv(f, header=f.tell()==0,index=False)


