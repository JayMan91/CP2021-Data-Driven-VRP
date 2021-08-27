import pandas as pd
from inspect import signature
import numpy as np
import glob
import itertools
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from MarkovModel import*
from  Full_model_padding import *
from VRP_LSTMmodel import *
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(filename='PaddedModel.log', 
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


test_days = [154,160, 166, 173, 180, 187, 194,
        155,161, 167, 174, 181, 188, 195,
        149,156, 168, 175, 182, 189, 196,
        150,162, 169, 176, 183, 190, 197,
        157,163, 170, 177, 184, 191, 198, 
        158, 164, 171, 178, 185, 192, 199,
        159, 165, 172, 179, 186, 193, 200]



net_list =  [MarkovwthStopembedding, NoWeek,MarkovwthStopembeddingwithLSTM, 
MarkovConvolution, NoHist, NoDist,NoMarkov,MarkovwthoutStopembedding,OnlyMarkov]
hyper_dict = {MarkovwthStopembedding:(50,0.1),
NoWeek:(50,0.1),
MarkovwthStopembeddingwithLSTM:(50,0.1),
MarkovConvolution:(100,0.01),
NoHist:(100,0.1),
NoDist:(100,0.1),NoMarkov:(100,0.1),
MarkovwthoutStopembedding:(100,0.1),OnlyMarkov:(100,0.1)}

lookback = 30
stop_embedding_size = 40
for net in net_list:
        lst = []
        epochs,lr = hyper_dict[net]
        for t in test_days:
                model =TwoStageVRP_padding(epochs= epochs,lookback_period=lookback,
                        lr=lr,net=  net, stop_embedding=True,n_features=1,
                        stop_embedding_size= stop_embedding_size)

                ev = model.evaluation(distance_mat, stops[:(t+1)],
                                        weekday[:(t+1)],n_vehicles[:(t+1)],
                                        opmat[:(t+1)], stop_wise_days,  demands[t], capacities[t:(t+1)],
                                        capacitated = True)
                lst.append({"Day":t,
                                "lookback":lookback,"stop_embedding_size":stop_embedding_size,
                                "epochs":epochs,"lr": lr,
                                "Model":str(net),
                                "bceloss":ev[2],
                                "training_bcelos":ev[3],
                                "Arc Difference":ev[0][0],"Arc Difference(%)":ev[0][1],
                                "Route Difference":ev[1][0],
                                "Route Difference(%)":ev[1][1],
                                "Distance":ev[4],
                                "Comment":ev[5] })
        df = pd.DataFrame(lst)
        filename = "NeuralNetModels.csv"
        with open(filename, 'a') as f:
                df.to_csv(f, header=f.tell()==0,index=False)

df = pd.read_csv("NeuralNetModels.csv")
df.Model = df.Model.map({"<class 'VRP_LSTMmodel.NoHist'>":"without past", 
"<class 'VRP_LSTMmodel.MarkovwthStopembedding'>":"Neural Net", 
"<class 'VRP_LSTMmodel.NoWeek'>":"without weekday",
"<class 'VRP_LSTMmodel.MarkovwthStopembeddingwithLSTM'>":"LSTM", 
"<class 'VRP_LSTMmodel.MarkovConvolution'>":"each stop different layer",
"<class 'VRP_LSTMmodel.NoDist'>":"without distance",
"<class 'VRP_LSTMmodel.NoMarkov'>":"without Markov",
"<class 'VRP_LSTMmodel.MarkovwthoutStopembedding'>":"without stop info",
"<class 'VRP_LSTMmodel.OnlyMarkov'>":"Only Markov"})
df.to_csv("NeuralNetModels.csv",index=False)