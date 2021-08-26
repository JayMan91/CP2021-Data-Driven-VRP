import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import re
from gurobipy import *
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax, log_softmax
dtype = torch.float
device = torch.device("cpu")
import logging
from MarkovModel import*
from VRP_model import *
from VRP_LSTMmodel import *
from Util import VRPGurobi, VRPsolutiontoList, eval_ad, eval_sd  
reluop = nn.ReLU() 
def pred0(output, target):
    # reluop = nn.ReLU()
    n_cols = target.shape[1]
    output_ = output.clone()
    for i in range(len(target)):
        
        n_ones =int( target[i].sum().item())
        # print(n_ones)
        v = torch.kthvalue(output_[i], n_cols- n_ones+1).values
        # print(v)
        output_[i][output_[i] >= v]=1
        output_[i][output_[i] < v]= 0
    
    # return reluop(target - output_).sum()
    return output_ #torch.mean((target - output_)**2)
def predother(output, target):
    output_ = output.clone()
    v = torch.topk(output,1).values
    output_[output_>=v]= 1
    output_[output_<v]= 0
    # return reluop(target - output).sum()
    return output_ #torch.mean((target - output)**2)


class TwoStageVRP_padding:
    def __init__(self,net,lookback_period =30,
        weekly=False,reverse=False,nnodes=74,
        embedding_size =6, nweekdays=7,n_features=2,
        optimizer=optim.Adam,epochs=20,stop_embedding=False,stop_embedding_size =10,
        **kwargs):
        self.net = net
        self.lookback_period = lookback_period
        self.weekly = weekly
        self.reverse = reverse
        self.nnodes = nnodes
        self.embedding_size = embedding_size
        self.epochs = epochs
      
        self.kwargs = kwargs
        self.stop_embedding = stop_embedding
        self.stop_embedding_size = stop_embedding_size
        self.n_features = n_features
        optim_args= [k for k, v in signature(optimizer).parameters.items()]
        self.optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}

        self.optimizer = optimizer

    def fit_predict(self,distance_mat,stops_list,weekday,n_vehicleslist,
        trgt, active_days,omega=1):
        '''
        past are extracted from the data,
        the data of day [t] is unused.

        this module also fits the models!
        '''
        trgt_past = trgt[:-1]
        stops_list_past = stops_list[:-1]
        weekday_past = weekday[:-1]
        n_vehicleslist_past = n_vehicleslist[:-1]
        activeindices = stops_list[-1]
        act = trgt[-1,:,:] 
        criterion = nn.BCELoss() 


        ###
        proba_mat = np.zeros((74,74))
        self.training_loss =0
        training_loss = []
        for test_stop in activeindices:
            model_stop =   self.net(self.embedding_size,
            self.lookback_period, 
            stop_embedding_size =self.stop_embedding_size, n_features=self.n_features) 
            optimizer_stop = self.optimizer(model_stop.parameters(),
            **self.optim_dict)

            stop_active_days = active_days[test_stop]
            till_day = np.searchsorted(stop_active_days ,len(trgt_past))
            stop_active_tilldays = stop_active_days[:till_day]
        
            if self.reverse:
                till_day = np.searchsorted(stop_active_days ,201-len(trgt_past))
                stop_active_tilldays = np.array(list(reversed(200 - stop_active_days[till_day:])))

            training_days = len(stop_active_tilldays)

            x_dist = log_softmax(distance_mat[test_stop]).astype(np.float32)
            y_train = np.zeros((training_days,
            self.nnodes)).astype(np.float32)
            x_train = np.zeros((training_days,
                self.lookback_period,self.nnodes)).astype(np.float32)
            x_mask = np.ones((training_days,self.nnodes)).astype(np.float32)
            x_vehicle = np.zeros((training_days,1)).astype(np.float32)
            x_stops = np.zeros((training_days,1)).astype(np.float32)
            x_week = np.zeros(training_days).astype(np.int64)
            x_markov = np.zeros((training_days,
            self.nnodes)).astype(np.float32)
            active_stops =[]


            for t in range(training_days):
                d = stop_active_tilldays[t]
                y_train[t] = trgt_past[d ][test_stop]
                lacti = stops_list_past[d ]
                x_vehicle[t] = n_vehicleslist_past[d]
                x_stops[t] = len(lacti)
                x_week[t] = weekday_past[d ]
                x_mask[t][lacti] = 0
                active_stops.append(lacti)

                if t >= self.lookback_period:
                    x_train[t] = trgt_past[stop_active_tilldays[(t-self.lookback_period):t],
                    test_stop]
                else:
                    x_train[t] = np.pad( trgt_past[stop_active_tilldays[:t],
                    test_stop],(( self.lookback_period - t,0),
                        (0,0)), 'constant', constant_values=(1/self.nnodes,0))

                
                markovmodel = MarkovCounter(test_stop,
                exp=0.7,beta=1,smoothing_value= 0.1,weekly= True)

                markovmodel.fit(trgt_past[stop_active_tilldays[:t]],
                weekday_past[stop_active_tilldays[:t]])
                x_markov[t] = markovmodel.predict(distance_mat,
                 stops_list_past[[d]], weekday_past[[d]],
                 n_vehicleslist_past[[d]],trgt_past[stop_active_tilldays[:d]])
            
            x_dist = torch.from_numpy(x_dist).to(device)
            x_markov = torch.from_numpy(x_markov).to(device)
            x_week = torch.from_numpy(x_week).to(device)
            x_train = torch.from_numpy(x_train).to(device)
            scaler = MinMaxScaler()
            x_stops = scaler.fit_transform(x_stops)
            if self.stop_embedding:
                x_features = x_vehicle
            else:
                x_features = np.concatenate((x_stops, x_vehicle), axis=1)


            x_features = torch.from_numpy(x_features).to(device)
            x_mask =  torch.from_numpy(x_mask).to(device)
            y_train = torch.from_numpy(y_train).to(device)
            # if test_stop ==0:
            #     criterion = my_loss0
            # else:
            #     criterion = my_loss_other
            # criterion = nn.NLLLoss() 

            for ep in range(self.epochs):
                optimizer_stop.zero_grad()
                if self.stop_embedding:
                    op = model_stop(active_stops, x_train,
                x_dist, x_features,x_markov, x_week,x_mask)
                else:
                    op = model_stop(x_train,
                x_dist, x_features,x_markov, x_week,x_mask)
                # loss = criterion( torch.log(op), torch.argmax(y_train, dim=1))
                if test_stop ==0:
                    y_pred = pred0(torch.exp(op),y_train)
                else:
                    y_pred = predother(torch.exp(op),y_train)
                
                # loss = -(op  * reluop(y_train -y_pred )).sum()/len(y_train)

                loss = -(op  * y_train).sum()/len(y_train)
                loss.backward()
                # print("Epochs: {} Loss: {}".format(ep,loss.item()))
                optimizer_stop.step()
            
            training_loss.append(loss.item())

            #### Training cplt
            ### Prediction phase
            model_stop.eval()
            eval_days= 1

            x_evaluation = np.zeros((eval_days,
                self.lookback_period,self.nnodes)).astype(np.float32)
            x_mask = np.ones((eval_days,self.nnodes)).astype(np.float32)
            x_vehicle = np.zeros((eval_days,1)).astype(np.float32)
            x_stops = np.zeros((eval_days,1)).astype(np.float32)
            x_week = np.zeros(eval_days).astype(np.int64)
            x_markov = np.zeros((eval_days,self.nnodes)).astype(np.float32)
            active_stops = []
            
            t = 0

            lacti = activeindices 
            active_stops.append(lacti)
            x_vehicle[t] = n_vehicleslist[-1]
            x_stops[t] = len(lacti)
            x_week[t] = weekday[-1 ]
            x_mask[t][activeindices] = 0

            if len(stop_active_tilldays) >= self.lookback_period:
                x_evaluation[t] = trgt[stop_active_tilldays[-(self.lookback_period):],
                test_stop]
            else:
                x_evaluation[t] = np.pad( trgt[stop_active_tilldays[-(self.lookback_period):],
                test_stop],(( self.lookback_period - len(stop_active_tilldays),0),
                    (0,0)), 'constant', constant_values=(1/self.nnodes,0))

            x_markov[t] = markovmodel.predict(distance_mat,
                stops_list[[-1]], weekday[[-1]],
                n_vehicleslist[[-1]],trgt_past)

            x_markov = torch.from_numpy(x_markov).to(device)
            x_week = torch.from_numpy(x_week).to(device)
            x_evaluation = torch.from_numpy(x_evaluation).to(device)
           
            x_stops = scaler.transform(x_stops)
            if self.stop_embedding:
                x_features = x_vehicle
            else:
                x_features = np.concatenate((x_stops, x_vehicle), axis=1)
            x_features = torch.from_numpy(x_features).to(device)
            x_mask =  torch.from_numpy(x_mask).to(device)

            if self.stop_embedding:
                predicted_proba = model_stop(active_stops, x_evaluation,
                x_dist, x_features,x_markov, x_week,x_mask).detach().numpy()
            else:
                predicted_proba = model_stop (x_evaluation,
            x_dist, x_features,x_markov, x_week,x_mask).detach().numpy()
            if omega==0:
                proba_mat[test_stop] = markovmodel.predict(distance_mat,
                stops_list[[-1]], weekday[[-1]],
                n_vehicleslist[[-1]],trgt_past)

            elif omega<1:
                exp = omega*np.exp(predicted_proba) + (1-omega)*np.exp(x_markov[t].detach().numpy())
                proba_mat[test_stop] =  np.log(exp)
            else:
                proba_mat[test_stop] = predicted_proba
            model_stop.train()
        self.training_loss = np.mean(training_loss)

        return proba_mat

    def evaluation(self,distance_mat,stops_list,weekday,n_vehicleslist,
        trgt, active_days,demands, capacities, capacitated =True,omega=1):
        '''
        demands and capacities are of day t
        rest are till day t
        this will be fed directly to predict, 
        which takes care of extracting the past

        '''
        trgt_past = trgt[:-1]
        # stops_list_past = stops_list[:-1]
        # weekday_past = weekday[:-1]
        # n_vehicleslist_past = n_vehicleslist[:-1]
        activeindices = stops_list[-1]
        act = trgt[-1,:,:] 
        
        proba_mat = self.fit_predict(distance_mat,stops_list,weekday,n_vehicleslist,
        trgt, active_days,omega)
        criterion = nn.NLLLoss()  #nn.BCELoss()
        bceloss = criterion( torch.from_numpy(proba_mat[activeindices,:][:,activeindices]),
            torch.from_numpy( np.argmax(act[activeindices,:][:,activeindices],axis=1)) ).item()

        
        proba_mat = -(proba_mat)
        '''
        the zeros come becuase of masking in the solution
        log of zero make it infinity, make this infinity with any scaler
        doesn't matter because the constraint specifies only active
        stops t be considered
        '''
        
        if capacitated:
            qcapacity = demands
            Q = capacities[0]            
        else:
            qcapacity = np.ones(74)
            Q = len(activeindices)
        solved,cmnt, sol,u = VRPGurobi(proba_mat,qcapacity,Q,
        n_vehicleslist[-1] ,activeindices)
        if solved:
            sol =  np.rint(sol)
            P = VRPsolutiontoList(sol)
            A = VRPsolutiontoList(act)
            # diff = act - sol

            # print(" Arc Difference {} {} Route Difference {}".format(np.sum( diff* (diff >0) ),
            #  eval_ad (P,A), eval_sd(P,A)))
        else:
            raise Exception("VRP not solved for day {}".format(len(trgt_past)))
        return eval_ad (P,A), eval_sd(P,A),bceloss,\
        self.training_loss,np.sum(distance_mat*sol), cmnt 


