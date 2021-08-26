import torch
from torch import nn, optim
import torch.nn.functional as f
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import re
from gurobipy import *
from MarkovModel import*
from Util import VRPGurobi, eval_ad, eval_sd,VRPsolutiontoList
import datetime
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax, log_softmax
from inspect import signature
from collections import OrderedDict
from csv import DictWriter
import os 
log1 = logging.getLogger('log1')
fileHandler = logging.FileHandler('Training.log', mode='w')
formatter = logging.Formatter( '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
log1.addHandler(fileHandler)
log1.setLevel(logging.INFO)
log1.info("this will be logged to file_1 ")


dtype = torch.float
device = torch.device("cpu")
import logging
log1 = logging.getLogger('log1')
fileHandler = logging.FileHandler('Training.log', mode='w')
formatter = logging.Formatter( '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
log1.addHandler(fileHandler)
log1.setLevel(logging.INFO)
log1.info("this will be logged to file_1 ")

class PredictOptimzeVRP:
    def __init__(self,
        net,  embedding_size =6, 
        nnodes=74, nweekdays=7,lookback_period=30,n_features=2,
        optimizer=optim.Adam,epochs=20,mu=1,
        capacitated =True,decision_focused=True,loss_type="Relu",loss_func="BB",
        stop_embedding=False,stop_embedding_size =10,relaxed=False, **kwargs):

        self.net = net
        self.stop_embedding = stop_embedding
        self.stop_embedding_size = stop_embedding_size
        self.nnodes = nnodes
        self.n_features = n_features
        self.lookback_period = lookback_period
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.mu = mu
        self.capacitated = capacitated
        self.decision_focused = decision_focused
        self.loss_type = loss_type
        self.loss_func = loss_func
        self.relaxed = relaxed
         
        optim_args= [k for k, v in signature(optimizer).parameters.items()]
        self.optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}

        self.optimizer = optimizer
      
    def fit(self,trgt,weekday,stops_list,n_vehicleslist,
        distance_mat,active_days,demands, capacities, *arg,**kwrg):
        '''
        everyting till t-1 i.e. [:t]

        '''

        training_days = len(trgt) - self.lookback_period

        x_past = np.zeros((training_days,self.nnodes,self.lookback_period,self.nnodes )).astype(np.float32)
        x_vehicle = np.zeros((training_days,1)).astype(np.float32)
        x_stops = np.zeros((training_days,1)).astype(np.float32)
        x_week = np.zeros(training_days).astype(np.int64)
        x_dist = log_softmax(distance_mat,axis=1).astype(np.float32)
        x_markov = np.zeros((training_days,
        self.nnodes, self.nnodes)).astype(np.float32)
        x_mask = np.ones((training_days,self.nnodes, self.nnodes)).astype(np.float32)

        self.mmodel_dict = {}
        for d in range(training_days):
            past = trgt[: (d + self.lookback_period)]
            lacti = stops_list[d + self.lookback_period]
            x_vehicle[d] = n_vehicleslist[d + self.lookback_period]
            x_stops[d] = len(lacti)
            x_week[d] = weekday[d + self.lookback_period]

            for stop in range(self.nnodes):
                if stop in lacti:
                    x_mask[d,stop][lacti] = 0
                stop_active_days = active_days[stop]
                till_day = np.searchsorted(stop_active_days ,len(past))
                stop_active_tilldays = stop_active_days[:till_day]
                if len(stop_active_tilldays) >= self.lookback_period:
                    # this stops has data for all lookback
                    x_past [d,stop] = past[stop_active_tilldays[-(self.lookback_period):],
                        stop]
                else:
                    x_past [d,stop] = np.pad( past[stop_active_tilldays[-(self.lookback_period):],
                        stop],(( self.lookback_period - len(stop_active_tilldays),0),
                        (0,0)), 'constant', constant_values=(1/self.nnodes,0))
                
                markovmodel = MarkovCounter(stop,
                exp=0.7,beta=1,smoothing_value= 0.1,weekly= True)

                markovmodel.fit(past[stop_active_tilldays],
                weekday[stop_active_tilldays])
                x_markov[d,stop] = markovmodel.predict(distance_mat,
                 stops_list[[(d + self.lookback_period)]],
                 weekday[[(d + self.lookback_period)]],
                 n_vehicleslist[[(d + self.lookback_period)]],past)
                if d== (training_days-1):
                    self.mmodel_dict[stop] = markovmodel

        logging.info("Date Prep Complete For Predict+Optimize")
        qcapacity = np.ones(74)
        Q = 74
        x_dist = torch.from_numpy(x_dist).to(device)
        x_markov = torch.from_numpy(x_markov).to(device)
        x_week = torch.from_numpy(x_week).to(device)
        x_past = torch.from_numpy(x_past).to(device)
        self.scaler = MinMaxScaler()
        x_stops = self.scaler.fit_transform(x_stops)
        x_features = np.concatenate((x_stops, x_vehicle), axis=1)
        if self.stop_embedding:
                x_features = x_vehicle
        else:
                x_features = np.concatenate((x_stops, x_vehicle), axis=1)

        x_features = torch.from_numpy(x_features).to(device)
        x_mask =  torch.from_numpy(x_mask).to(device)

        self.model_dict= {}
        self.optimizer_dict = {}
        for stop in range(self.nnodes):
            model =   self.net(self.embedding_size,
            self.lookback_period,stop_embedding_size =self.stop_embedding_size,
            decision_focused= self.decision_focused, n_features=self.n_features)

            optimizer = self.optimizer(model.parameters(),
            **self.optim_dict)
            self.model_dict[stop] = model
            self.optimizer_dict[stop] = optimizer
        logging.info("Model and Optimizer setup done")
        training_loss = []

        for ep in range(self.epochs):
            print("####",ep +1, "Epoch started #######",)

            pred_dict ={}
            self.total_ad = 0
            total_ad =[]
            for d in range(training_days):
                lacti = stops_list[d + self.lookback_period]
                proba_mat_np = np.zeros((self.nnodes,self.nnodes))
                logging.info("Training for day {} begins".format(d))
                for stop in range(self.nnodes):
                    model = self.model_dict[stop]
                    if self.stop_embedding:
                        pred_dict[stop] = (model([lacti],x_past[[d],stop,:,:],
                    x_dist[stop], x_features[[d]],x_markov[[d], stop,:], 
                    x_week[[d]],x_mask[[d], stop,:]))
                    else:
                        pred_dict[stop] = (model(x_past[[d],stop,:,:],
                    x_dist[stop], x_features[[d]],x_markov[[d], stop,:], 
                    x_week[[d]],x_mask[[d], stop,:]))
                                        
                    proba_mat_np[stop] = pred_dict[stop].detach().numpy()
                # logging.info("probability matrix populated")

                actual = trgt[d + self.lookback_period]
                '''
                Decision Focused
                '''
                ## proba_mat_np is matrix of negative logprobs with proper masking
                if self.capacitated:
                    qcapacity = demands [d + self.lookback_period]
                    Q = capacities[d + self.lookback_period] 
                # print("day", d + self.lookback_period,"capacity check, number of vehciles",n_vehicleslist[(d + self.lookback_period) ],
                # sum(qcapacity), Q)
                if self.loss_func=="BB":
                    solved,cmnt, sol_hat,u = VRPGurobi(proba_mat_np,qcapacity,Q,
                    n_vehicleslist[(d + self.lookback_period) ] ,
                    stops_list[d + self.lookback_period], relaxed= self.relaxed )
                    if solved:
                        sol_hat =  np.rint(sol_hat)
                        if self.loss_type== "Relu":
                            del_l = np.where(actual > sol_hat, -1, 0)
                        elif self.loss_type== "Squared":
                            del_l = sol_hat - actual
                        else:
                            raise Exception("Unknown Loss type")

                        proba_tilde = proba_mat_np + self.mu*del_l

                        solved,cmnt, sol_tilde,u = VRPGurobi(proba_tilde,qcapacity,Q,
                    n_vehicleslist[(d + self.lookback_period) ] ,
                    stops_list[d + self.lookback_period], relaxed= self.relaxed )
                        sol_tilde =  np.rint(sol_tilde)
                        grad = torch.from_numpy((sol_tilde - sol_hat)/self.mu).float()
                if self.loss_func=="SPO":
                    solved,cmnt, sol_hat,u = VRPGurobi(proba_mat_np+self.mu,qcapacity,Q,
                    n_vehicleslist[(d + self.lookback_period) ] ,
                    stops_list[d + self.lookback_period] )
                    if solved:
                        sol_hat =  np.rint(sol_hat)
                        grad = torch.from_numpy((actual - (sol_hat))).float()
                for stop in lacti:#lacti range(self.nnodes)
                    optimizer =  self.optimizer_dict[stop]
                    optimizer.zero_grad()
                    pred_dict[stop].backward(grad[stop].view(1,-1),
                        retain_graph=True)
                    optimizer.step()
                        # for p in self.model_dict[2].parameters():
                        #     pass
                        # print(p.data)
                diff = actual - sol_hat
                log1.info("Epoch {} Day {} Arc Diff {}".format(ep,d,
                np.sum( diff* (diff >0) )))
                training_loss.append({"Mu":self.mu,
                "Epoch":ep,"Day":d + self.lookback_period,
                "Arc Difference":np.sum(diff* (diff >0))})
                # self.total_ad += np.sum(diff* (diff >0))
                total_ad.append(np.sum(diff* (diff >0)))
        self.total_ad = np.mean(total_ad)
        keys = training_loss[0].keys()
        filename = "DecisionFocusedTraining.csv"
        if os.path.exists(filename):
            os.remove(filename) 
        with open(filename, "w") as f:
            dict_writer = DictWriter(f, keys, delimiter=",")
            dict_writer.writeheader()
            for value in training_loss:
                dict_writer.writerow(value)
        
    def predict(self,distance_mat,stops_list,weekday,n_vehicleslist,past,
     active_days,  *arg,**kwrg):
        '''
        One instance
        past from 0 to t-1 i.e :t
        other [t]
        '''

        prediction_days = 1

        x_past = np.zeros((prediction_days,self.nnodes,self.lookback_period,self.nnodes )).astype(np.float32)
        x_vehicle = np.zeros((prediction_days,1)).astype(np.float32)
        x_stops = np.zeros((prediction_days,1)).astype(np.float32)
        x_week = np.zeros(prediction_days).astype(np.int64)
        x_dist = log_softmax(distance_mat,axis=1).astype(np.float32)
        x_markov = np.zeros((prediction_days,
        self.nnodes, self.nnodes)).astype(np.float32)
        x_mask = np.ones((prediction_days,self.nnodes, self.nnodes)).astype(np.float32)


        # past = past[-( self.lookback_period):]
        lacti = stops_list[ 0]
        x_vehicle[0] = n_vehicleslist[ 0]
        x_stops[0] = len(lacti)
        x_week[0] = weekday[ 0]

        for stop in range(self.nnodes):
            if stop in lacti:
                x_mask[0,stop][lacti] = 0
            stop_active_days = active_days[stop]
            till_day = np.searchsorted(stop_active_days ,len(past))
            stop_active_tilldays = stop_active_days[:till_day]
            if len(stop_active_tilldays) >= self.lookback_period:
                # this stops has data for all lookback
                x_past [0,stop] = past[stop_active_tilldays[-(self.lookback_period):],
                    stop]
            else:
                x_past [0,stop] = np.pad( past[stop_active_tilldays[-(self.lookback_period):],
                    stop],(( self.lookback_period - len(stop_active_tilldays),0),
                    (0,0)), 'constant', constant_values=(1/self.nnodes,0))
            markovmodel = self.mmodel_dict[stop]

            x_markov[0,stop] = markovmodel.predict(distance_mat,
                stops_list,
                weekday,
                n_vehicleslist,past)
        
        logging.info("Date Prep Complete For Predicting")
        x_dist = torch.from_numpy(x_dist).to(device)
        x_markov = torch.from_numpy(x_markov).to(device)
        x_week = torch.from_numpy(x_week).to(device)
        x_past = torch.from_numpy(x_past).to(device)

        x_stops = self.scaler.transform(x_stops)
        if self.stop_embedding:
                x_features = x_vehicle
        else:
                x_features = np.concatenate((x_stops, x_vehicle), axis=1)

        x_features = torch.from_numpy(x_features).to(device)
        x_mask =  torch.from_numpy(x_mask).to(device)  

        proba_mat = torch.zeros(self.nnodes,self.nnodes)

        for stop in range(self.nnodes):
            model = self.model_dict[stop]
            model.eval()
            # proba_mat[stop,:]= f.normalize( torch.exp(-model(x_past[:,stop,:,:],
            # x_dist[stop], x_features,x_markov[:, stop,:], 
            # x_week,x_mask[:, stop,:])) , p=1, dim=1)

            if self.stop_embedding:
                proba_mat[stop,:]= model([lacti], x_past[:,stop,:,:],
            x_dist[stop], x_features,x_markov[:, stop,:], 
            x_week,x_mask[:, stop,:])
            else:
               proba_mat[stop,:]= model(x_past[:,stop,:,:],
            x_dist[stop], x_features,x_markov[:, stop,:], 
            x_week,x_mask[:, stop,:])


            model.train()

        return proba_mat.detach().numpy()


    def evaluation(self,distance_mat,stops_list,weekday,n_vehicleslist,
        trgt, active_days,demands, capacities, capacitated =True):
        '''
        trgt contains both past and current
        that is [:(t+1)]
        This will call the predict function.
        Predict takes the past and current contextuals.
        '''
        past = trgt[:-1,:,:]
        act = trgt[-1,:,:] 
        activeindices = stops_list[0]
        if capacitated:
            qcapacity = demands
            Q = capacities[0]            
        else:

            qcapacity = np.ones(74)
            Q = len(activeindices)
        proba_mat = self.predict(distance_mat,stops_list,
        weekday,n_vehicleslist,past, active_days)
        # criterion = nn.BCELoss()
        # bceloss = criterion( torch.from_numpy(proba_mat[activeindices,:][:,activeindices]).float(),
        #     torch.from_numpy(act[activeindices,:][:,activeindices]).float() ).item()

        criterion = nn.NLLLoss()  #nn.BCELoss()
        bceloss = criterion( torch.from_numpy(-proba_mat[activeindices,:][:,activeindices]),
            torch.from_numpy( np.argmax(act[activeindices,:][:,activeindices],axis=1)) ).item()

        solved,cmnt, sol,u = VRPGurobi(proba_mat,qcapacity,Q,
        n_vehicleslist[-1] ,activeindices)
        if solved:
            sol =  np.rint(sol)
            P = VRPsolutiontoList(sol)
            A = VRPsolutiontoList(act)
        else:
            raise Exception("VRP not solved for day {}".format(len(past)))
        return eval_ad (P,A), eval_sd(P,A),bceloss,\
        self.total_ad, np.sum(distance_mat*sol), cmnt         