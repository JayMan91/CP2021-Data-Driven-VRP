import pandas as pd
import numpy as np
from scipy.special import softmax
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from Util import VRPGurobi, VRPsolutiontoList, eval_ad, eval_sd  
'''
With week day
'''
class TwoStageVRP:
    def __init__(self,model=None,lookback_period =30,
        weekly=False,reverse=False,nnodes=74,**kwargs):
        self.model = model
        self.lookback_period = lookback_period
        self.weekly = weekly
        self.reverse = reverse
        self.nnodes = nnodes
        self.kwargs = kwargs
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

        ###
        proba_mat = np.zeros((74,74))
        self.training_loss = 0
        for test_stop in activeindices:
            stop_active_days = active_days[test_stop]
            till_day = np.searchsorted(stop_active_days ,len(trgt_past))
            stop_active_tilldays = stop_active_days[:till_day]
            
        
            lookback_period= min(self.lookback_period,len(stop_active_tilldays)//2)
            clf = self.model(training_stops= test_stop,
                lookback_period = lookback_period,weekly= self.weekly, **self.kwargs)

            #     mmmodel = MarkovCounter(beta=1, training_stops= test_stop,
            #         exp=0.7,smoothing_value=0.1, weekly=True )
            # else:
            #     # clf = MarkovCounter(beta=0, training_stops= test_stop    )
            #     clf = MarkovCounter(beta=1, training_stops= test_stop,
            #         exp=0.7,smoothing_value=0.1, weekly=True )
            #     mmmodel = MarkovCounter(beta=0, training_stops= test_stop,
            #          weekly=True )

            clf.fit(trgt_past[ stop_active_tilldays ],
                weekday_past[stop_active_tilldays], 
                stops_list_past[stop_active_tilldays],
                n_vehicleslist_past[stop_active_tilldays],distance_mat)
            
            ###########   training loss    ####################
            pred_train = clf.predict(distance_mat,
            stops_list_past[stop_active_tilldays][lookback_period:],
            weekday_past[stop_active_tilldays][lookback_period:],
            n_vehicleslist_past[stop_active_tilldays][lookback_period:],
            trgt_past[ stop_active_tilldays ])
            act_train = trgt_past[stop_active_tilldays][lookback_period:,test_stop]

            criterion = nn.NLLLoss() # nn.BCELoss()
            self.training_loss += criterion( torch.from_numpy(pred_train),
            torch.from_numpy(np.argmax(act_train,axis=1) ) ).item()


            predicted_proba = clf.predict(distance_mat,stops_list[[-1]],
            weekday[[-1]],n_vehicleslist[[-1]],trgt_past[ stop_active_tilldays ])
            # proba_mat[test_stop] = predicted_proba

            # markov_proba = mmmodel.predict(distance_mat,stops_list[[-1]],
            # weekday[[-1]],n_vehicleslist[[-1]],trgt_past[stop_active_tilldays ])
            proba_mat[test_stop] = predicted_proba 
        return proba_mat # log proba

    def evaluation(self,distance_mat,stops_list,weekday,n_vehicleslist,
        trgt, active_days,demands, capacities, capacitated =True, omega=1):
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
        else:
            raise Exception("VRP not solved for day {}".format(len(trgt_past)))
        return eval_ad (P,A), eval_sd(P,A),bceloss,\
        self.training_loss,np.sum(distance_mat*sol), cmnt 
        # 0,0: absolute arc difference
        # 0,1: percent arc difference
        # 1,0: absolute route difference
        # 1,1: percent route difference
        # 2: cmnt


class MarkovCounter:
    def __init__(self,training_stops, exp=0.01,beta=0.2,smoothing_value= 0.1,
        weekly=False,*arg,**kwrg):
        self.training_stops = training_stops
        self.exp = exp # exponential smoothing
        self.beta = beta
        self.smoothing_value = smoothing_value
        self.weekly = weekly
        
    def fit(self, opmat_train,weeklist,*arg,**kwrg):     
        # compute proba matrix M
        opmat_train_ = opmat_train[:,
        self.training_stops:(self.training_stops+1)]
        
        assert len(opmat_train_) == len(weeklist)
        _,r,c = opmat_train_.shape
        if self.weekly:
            self.M = np.zeros([7,r,c])
            for w in range(7):
                opmat_sub = opmat_train_[np.where(weeklist==w)]
                for i in range(len(opmat_sub)):
                    wt = self.exp*(1-self.exp)**(len(opmat_sub)-(i+1))
                    self.M[w] += wt*opmat_sub[i]
        else:
            self.M = np.zeros([r,c])
            for i in range(len(opmat_train_)):
                    wt = self.exp*(1-self.exp)**(len(opmat_train_)-(i+1))
                    self.M += wt*opmat_train_[i]


    def predict(self,distance_mat, inst,weekday,
    n_vehicleslist,past,proba=True, full=True,*arg,**kwrg):
        # only distance_mat, inst and week day are required
       
        # returns list of proba for inst
        assert len(inst)==len(weekday)
        days = len(inst)
        y = np.zeros((days,distance_mat.shape[1]))

        if self.beta<1:
            R = np.exp(-distance_mat[[self.training_stops],:]) # reduced distance_mat
            # if not full:
            #     R = R[np.ix_(inst_,inst_)]
            # convert to probabilities
            R[R == 1] = np.min(R)
            R = R/R.sum(axis=-1, keepdims=True) 

        for d in range(days):
            if self.beta>0:
                weekday_ = weekday[d]
                inst_ = inst[d]
                if self.weekly:
                    assert weekday_ is not None
                    M_week = self.M[weekday_]
                else:
                    M_week = self.M
                M_week = M_week + self.smoothing_value
                
                raw_M = (M_week/M_week.sum(axis=-1, keepdims=True))
                # raw_M = softmax(M_week,axis=-1)
                
                # if not full:
                #     raw_M = raw_M[np.ix_(inst_,inst_)]
            if self.beta==0:
                raw_M = R
            elif self.beta<1:
                raw_M = self.beta*raw_M + (1- self.beta)*R
            if not full:
                raw_M = raw_M[np.ix_(inst_,inst_)]
            if proba:
                y [d] = np.log(raw_M)
            if not proba:
                y = np.argmax(y,dim=1)
     
        return y
