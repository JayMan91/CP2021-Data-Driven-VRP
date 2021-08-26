import pandas as pd
import numpy as np
import glob
import re
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from gurobipy import *
import datetime
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from inspect import signature
dtype = torch.float
device = torch.device("cpu")
import logging
import warnings
## data loading
'''
separate model for each stop
'''
# npzfile = np.load("daily_stops.npz",allow_pickle=True,)
# stops = npzfile['stops_list'] # 201 length list induicating which stops are active for each day
# n_vehicles= npzfile['nr_vehicles'] #n_vehicles for each day
# weekday= npzfile['weekday'] # categorical input
# npzfile = np.load("daily_routematrix.npz", allow_pickle=True)
# opmat = npzfile['incidence_matrices'] # solutions for each day as an incidence matrix
# nxt_stops  = npzfile['next_stops']
# distance_mat = np.load("Distancematrix.npy")
# node_mat = np.load("node_category.npy")

def make_onehot(vec,num_classes):
    vec_ =  vec.reshape(len(vec),1)

    one_hot_target = (vec_ == torch.arange(num_classes).reshape(1, num_classes)).float()
    return one_hot_target


n_stops = 74
training_stops = 60
## classifier model
class costnet(nn.Module):
    def __init__(self,embedding_size,n_features=2,
                 nnodes= 74, nweekdays=7,onehot=True,**kwargs):
        super().__init__()

        if onehot:
            self.linear = nn.Linear(nnodes+nweekdays+n_features,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.linear = nn.Linear(nnodes+embedding_size+n_features,nnodes)
    
        self.nnodes = nnodes
        self.nweekdays = nweekdays
        self.onehot = onehot
    def forward(self,x_dist, x_features,x_week,x_mask):

        if self.onehot:
            x_emb =  make_onehot(x_week,self.nweekdays)
        else:
            x_emb = self.embeddings(x_week)

        n_batches = len(x_week)
        x = torch.cat([ x_emb, x_features, x_dist.expand(n_batches,-1)], 1)

        x = self.linear(x)
        m = nn.Softmax(dim=1)
        return m(x)* x_mask

class costnetResidual(nn.Module):
    def __init__(self, embedding_size=1,n_features=2,
                 nnodes= 74, nweekdays=7,onehot=True,**kwargs):
        super().__init__()
        if onehot:
            self.linear = nn.Linear(nweekdays+n_features,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.linear = nn.Linear(embedding_size+n_features,nnodes)
        self.nnodes = nnodes
        self.onehot = onehot
        self.nweekdays = nweekdays
    def forward(self,x_dist, x_features,x_week,x_mask):
        if self.onehot:
            x_emb =  make_onehot(x_week,self.nweekdays)
        else:
            x_emb = self.embeddings(x_week)
        x = torch.cat([ x_emb, x_features], 1)

        x = self.linear(x) - x_dist
        m = nn.Softmax(dim=1)
        return m(x)* x_mask




class VRPNet:
    def __init__(self,training_stops,
                 net=costnet,
                 embedding_size =2,nnodes=74,n_features=2, nweekdays=7,
                 optimizer=optim.Adam,epochs=20,onehot=False,**kwargs):
        
        self.net = net
        self.training_stops = training_stops
        self.nnodes = nnodes

        self.model = self.net(embedding_size= embedding_size,
            nnodes=nnodes,n_features=n_features, nweekdays=7,onehot=onehot)
        self.epochs = epochs
        
        optim_args= [k for k, v in signature(optimizer).parameters.items()]
        optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}

        self.optimizer = optimizer(self.model.parameters(), **optim_dict)


    def fit(self,trgt,weekday,stops_list,n_vehicleslist,distance_mat, *arg,**kwrg):
        n_days = len(stops_list)
        x_mask = np.zeros((n_days,self.nnodes)).astype(np.float32)
        x_vehicle = np.zeros((n_days,1)).astype(np.float32)
        x_stops = np.zeros((n_days,1)).astype(np.float32)
        x_week = np.zeros(n_days).astype(np.int64)
        xdist = softmax(distance_mat[self.training_stops]).astype(np.float32)
        y_train = np.zeros((n_days,self.nnodes)).astype(np.float32)
        
        cnt = 0
        for d in range(n_days):
            lacti = stops_list[d]
            if self.training_stops not in lacti:
                raise Exception('Stop should be in the training days')
                # print("Stops {} Not in day {}".format(self.training_stops,
                #     d))
            else:
                x_vehicle[d] = n_vehicleslist[d]
                x_stops[d] = len(lacti)
                x_week[d] = weekday[d]
                x_mask[d][lacti] =1
                y_train[d] = trgt[d][self.training_stops]
                cnt +=1
        x_mask =  torch.from_numpy(x_mask[:cnt]).to(device)
        self.scaler = MinMaxScaler()
        x_stops = self.scaler.fit_transform(x_stops)        
        x_features = np.concatenate((x_stops, x_vehicle), axis=1)
        # x_features = np.zeros((n_days,0)).astype(np.float32)



        x_features = torch.from_numpy(x_features[:cnt]).to(device)
        x_week = torch.from_numpy(x_week[:cnt]).to(device)
        xdist = torch.from_numpy(xdist).to(device)
        y_train = torch.from_numpy(y_train[:cnt]).to(device)
        
        criterion = nn.BCELoss() #nn.BCEWithLogitsLoss()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            op = self.model(xdist, x_features,x_week,x_mask)
            loss = criterion(op, y_train)
            loss.backward()
            self.optimizer.step()
            print("Epochs: {} Loss: {}".format(epoch,loss.item()))


    def predict(self,distance_mat,stops_list,weekday,n_vehicleslist,past=None, proba=True, *arg,**kwrg):
        n_days = len(stops_list)
        x_mask = np.zeros((n_days,self.nnodes)).astype(np.float32)
        x_vehicle = np.zeros((n_days,1)).astype(np.float32)
        x_stops = np.zeros((n_days,1)).astype(np.float32)
        x_week = np.zeros(n_days).astype(np.int64)
        xdist = softmax(distance_mat[self.training_stops]).astype(np.float32)
        y = np.zeros((n_days,self.nnodes)).astype(np.float32)
        
        cnt = 0
        active_days =[]
        for d in range(n_days):
            lacti = stops_list[d]
            if self.training_stops not in lacti:
                warnings.warn('Stop {} not in the evaluation days, \n predicted probabilities would be zero'.format(self.training_stops))

            else:
                x_vehicle[d] = n_vehicleslist[d]
                x_stops[d] = len(lacti)
                x_week[d] = weekday[d]
                x_mask[d][lacti] =1
                cnt +=1
                active_days.append(d)
        x_mask =  torch.from_numpy(x_mask[:cnt]).to(device)
        x_stops = self.scaler.transform(x_stops)
        x_features = np.concatenate((x_stops, x_vehicle), axis=1)

        x_features = torch.from_numpy(x_features[:cnt]).to(device)        
        # x_vehicle = torch.from_numpy(x_vehicle[:cnt]).to(device)
        x_week = torch.from_numpy(x_week[:cnt]).to(device)
        xdist = torch.from_numpy(xdist).to(device)
        
        self.model.eval()
        op = self.model(xdist, x_features,x_week,x_mask)
        # m = nn.Softmax(dim=1)
        # op = m(op)
       
        self.model.train()
        y[active_days] = op.detach().numpy()
        if not proba:
            y = torch.argmax(y,dim=1)
        return y
    # def predict(self,stops_list,n_vehicleslist,weekday,xdist):
    #     raise Exception('Not Implemented')
class VRPresidualNet(VRPNet):
    def __init__(self,training_stops,
                 net=costnetResidual,
                 embedding_size =2,nnodes=74,n_features=2,nweekdays=7,
                 optimizer=optim.Adam,epochs=20,onehot=False, **kwargs):
        
        self.net = net
        self.training_stops = training_stops
        self.nnodes = nnodes

        self.model = self.net(embedding_size= embedding_size,
            nnodes=nnodes,n_features=n_features, nweekdays=7,onehot= onehot)
        self.epochs = epochs
        optim_args= [k for k, v in signature(optimizer).parameters.items()]
        optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}

        self.optimizer = optimizer(self.model.parameters(), **optim_dict)

   

