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
import datetime
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax, log_softmax
from inspect import signature
from collections import OrderedDict
dtype = torch.float
device = torch.device("cpu")
import logging
import warnings
def make_onehot(vec,num_classes):
    vec_ =  vec.reshape(len(vec),1)

    one_hot_target = (vec_ == torch.arange(num_classes).reshape(1, num_classes)).float()
    return one_hot_target

# class LSTMpred(nn.Module):
#     def __init__(self,input_dim, hidden_size, num_layers, target_dim,
#     embedding_size,lookback_period, nnodes= 74,n_features=2, 
#     weekly=False,nweekdays=7,drop_prob=0.01,
#     **kwargs):
#         super().__init__()

#         self.weekly= weekly        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.embeddings = nn.Embedding(nweekdays, embedding_size)
#         self.lstm = nn.LSTM(input_dim,hidden_size, num_layers, 
#         	batch_first=True, dropout=drop_prob)
#         self.droput = nn.Dropout(p= drop_prob)
#         if weekly:
#             self.fc = nn.Linear(n_features,target_dim)
#         else:
#             self.fc = nn.Linear(embedding_size+n_features,target_dim)
#         self.nnodes = nnodes
#         self.lookback_period =lookback_period
#     def forward(self,x, h, x_dist, x_features,x_week,x_mask):
#         m = nn.Softmax(dim=1)
        
#         out, h = self.lstm(x, h)
#         out = out[:,-1,:]# (*,74)
        
#         if self.weekly:
#             x_ = x_features
#         else:
#             x_emb = self.embeddings(x_week)
#             x_ = torch.cat([ x_emb, x_features], 1)      
#         x_ = (self.fc(x_)+ out - x_dist)
#         return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)

# class LSTMFullpred(nn.Module):
#     def __init__(self,input_dim, hidden_size, num_layers, target_dim,
#     embedding_size,lookback_period,nnodes= 74,n_features=2,
#     weekly=False, nweekdays=7,drop_prob=0.01,
#     **kwargs):
#         super().__init__()

#         self.weekly= weekly        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.embeddings = nn.Embedding(nweekdays, embedding_size)

#         self.lstm = nn.LSTM(input_dim,hidden_size, num_layers, 
#         	batch_first=True, dropout=drop_prob)
#         self.droput = nn.Dropout(p= drop_prob)

#         # self.fc = nn.Linear(embedding_size+n_features,target_dim)
#         if weekly:
#             self.fc = nn.Linear(n_features + nnodes,target_dim)
#         else:
#             self.fc = nn.Linear(embedding_size+n_features + nnodes,target_dim)
#         self.nnodes = nnodes
#         self.lookback_period =lookback_period
#     def forward(self,x, h, x_dist, x_features,x_week,x_mask):
#         n_rows = x_features.shape[0]
#         m = nn.Softmax(dim=1)
        
#         out, h = self.lstm(x, h)
#         out = out[:,-1,:]# (*,74)
        
#         if self.weekly:
#             x_ = torch.cat([  x_features, x_dist.expand(n_rows,-1)], 1)
#         else:
#             x_emb = self.embeddings(x_week)
#             x_ = torch.cat([ x_emb, x_features, x_dist.expand(n_rows,-1)], 1)           
#         x_ = (self.fc(x_)+ out)

#         return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)

class MarkovConvolution(nn.Module):
    def __init__(self, embedding_size,
        lookback_period,stop_embedding_size=12, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        if weekly:
            self.conv = nn.Conv1d(1,nnodes,n_features+3+stop_embedding_size,
            n_features+3+stop_embedding_size)       
        elif onehot:
            self.conv = nn.Conv1d(1,nnodes,
           nweekdays+n_features+stop_embedding_size+3,nweekdays+n_features+stop_embedding_size+3)
        else:
            self.conv = nn.Conv1d(1,nnodes,embedding_size+n_features+3+stop_embedding_size,
            embedding_size+n_features+3+stop_embedding_size)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)

        self.weekly= weekly      
        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)  
        
        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p= drop_prob)        
        self.fc1 = nn.Linear(lookback_period,1)
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,stops,x, x_dist, x_features,x_markov, x_week,x_mask):
        x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))

        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2))


        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)

        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        x_embed.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        out,x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)], -1)

        n_f = x_.shape[-1]
        x_ =  self.conv(x_.view(-1,1,self.nnodes*n_f ))
        x_ = torch.diagonal(x_,  dim1=1, dim2=2)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            # return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)
            # return m(x_)
class MarkovwthStopembedding(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, stop_embedding_size=12,target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p= drop_prob)        
        self.fc1 = nn.Linear(lookback_period,1)
        if weekly:
            self.fc2 = nn.Linear(n_features+3+stop_embedding_size,1)       
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features+3+stop_embedding_size,1)
        else:
            self.fc2 = nn.Linear(embedding_size+n_features+3+stop_embedding_size,1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)        
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,stops,x, x_dist, x_features,x_markov, x_week,x_mask):
        x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))
        
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2))
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)
        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        x_embed.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        out,x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)], -1)
        n_f = x_.shape[-1]
        x_ =  self.fc2(x_).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)


class MarkovwthoutStopembedding(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, stop_embedding_size=12,target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p= drop_prob)        
        self.fc1 = nn.Linear(lookback_period,1)
        if weekly:
            self.fc2 = nn.Linear(n_features+3,1)       
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features+3,1)
        else:
            self.fc2 = nn.Linear(embedding_size+n_features+3,1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
        # self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)        
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,stops,x, x_dist, x_features,x_markov, x_week,x_mask):
        # x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))
        
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2))
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)
        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        # x_embed.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        out,x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)], -1)
        n_f = x_.shape[-1]
        x_ =  self.fc2(x_).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)


class MarkovwthStopembeddingwithLSTM(nn.Module):
    def __init__(self, embedding_size,
        lookback_period,num_layers=1, stop_embedding_size=12,target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers

        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p= drop_prob)        
        # self.fc1 = nn.Linear(lookback_period,1)
        self.lstm = nn.LSTM(nnodes,nnodes, num_layers, 
        	batch_first=True)


        if weekly:
            self.fc2 = nn.Linear(n_features+3+stop_embedding_size,1)       
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features+3+stop_embedding_size,1)
        else:
            self.fc2 = nn.Linear(embedding_size+n_features+3+stop_embedding_size,1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)        
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,stops,x, x_dist, x_features,x_markov, x_week,x_mask):
        x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))
        
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        # out = self.fc1(x.transpose(1,2))
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.nnodes)) 
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.nnodes))
        out, h = self.lstm(x, (h_0,c_0))
        out = out[:,-1,:].unsqueeze(-1)


        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)
        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        x_embed.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        out,x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)], -1)
        n_f = x_.shape[-1]
        x_ =  self.fc2(x_).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)


class MarkovLinear(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()


        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p= drop_prob)        
        self.fc1 = nn.Linear(lookback_period,1)
        if weekly:
            self.fc2 = nn.Linear(n_features+3,1)       
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features+3,1)
        else:
            self.fc2 = nn.Linear(embedding_size+n_features+3,1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)        
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,x, x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2))
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)
        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        out,x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)], -1)
        n_f = x_.shape[-1]
        x_ =  self.fc2(x_).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)


class NoHist(nn.Module):
    def __init__(self, embedding_size,
        lookback_period,stop_embedding_size=10, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p= drop_prob)        

        if weekly:
            self.fc2 = nn.Linear(n_features+2+stop_embedding_size,1)       
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features+2+stop_embedding_size,1)
        else:
            self.fc2 = nn.Linear(embedding_size+n_features+2+stop_embedding_size,1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)        
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,stops,x, x_dist, x_features,x_markov, x_week,x_mask):
        x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))
        
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)
        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        x_embed.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)], -1)
        n_f = x_.shape[-1]
        x_ =  self.fc2(x_).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)

# class NoHist(MarkovLinear):
#     def __init__(self, embedding_size,
#         lookback_period, target_dim=1,nnodes= 74,n_features=2, 
#         nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
#         decision_focused=False, **kwargs):
#         super().__init__(embedding_size,
#         lookback_period, target_dim,nnodes,n_features, 
#         nweekdays,onehot, drop_prob,weekly,
#         decision_focused, **kwargs)


#         if weekly:
#             self.fc2 = nn.Linear(n_features+2,1)       
#         elif onehot:
#             self.fc2 = nn.Linear(nweekdays+n_features+2,1)
#         else:
#             self.fc2 = nn.Linear(embedding_size+n_features+2,1)
#             self.embeddings = nn.Embedding(nweekdays, embedding_size)        


#     def forward(self,x, x_dist, x_features,x_markov, x_week,x_mask):
#         n_rows = x_features.shape[0]
#         m = nn.Softmax(dim=1)
#         if self.weekly:
#             x_ = x_features
#         else:
#             if self.onehot:
#                 x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
#             else:
#                 x_emb = self.embeddings(x_week) #(*,embedding_size)        
#             x_ = torch.cat([ x_emb, x_features], 1)
#         x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
#         x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
#         x_markov.unsqueeze(-1)], -1)
#         n_f = x_.shape[-1]
#         x_ =  self.fc2(x_).squeeze(-1)

#         if self.decision_focused:
#             mlog = nn.LogSoftmax(dim=1)
#             return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
#         else:
#             mlog = nn.LogSoftmax(dim=1)
#             return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)

class NoDist(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, stop_embedding_size=12,target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p= drop_prob)        
        self.fc1 = nn.Linear(lookback_period,1)
        if weekly:
            self.fc2 = nn.Linear(n_features+2+stop_embedding_size,1)       
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features+2+stop_embedding_size,1)
        else:
            self.fc2 = nn.Linear(embedding_size+n_features+2+stop_embedding_size,1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)        
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,stops,x, x_dist, x_features,x_markov, x_week,x_mask):
        x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))
        
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2))
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)
        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        x_embed.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        out,x_markov.unsqueeze(-1)], -1)
        n_f = x_.shape[-1]
        x_ =  self.fc2(x_).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)


# class NoDist(MarkovLinear):
#     def __init__(self, embedding_size,
#         lookback_period, target_dim=1,nnodes= 74,n_features=2, 
#         nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
#         decision_focused=False, **kwargs):
#         super().__init__(embedding_size,
#         lookback_period, target_dim,nnodes,n_features, 
#         nweekdays,onehot, drop_prob,weekly,
#         decision_focused, **kwargs)
#         if weekly:
#             self.fc2 = nn.Linear(n_features+2,1)       
#         elif onehot:
#             self.fc2 = nn.Linear(nweekdays+n_features+2,1)
#         else:
#             self.fc2 = nn.Linear(embedding_size+n_features+2,1)
#             self.embeddings = nn.Embedding(nweekdays, embedding_size)        


#     def forward(self,x, x_dist, x_features,x_markov, x_week,x_mask):
#         n_rows = x_features.shape[0]
#         m = nn.Softmax(dim=1)
#         out = self.fc1(x.transpose(1,2))
#         if self.weekly:
#             x_ = x_features
#         else:
#             if self.onehot:
#                 x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
#             else:
#                 x_emb = self.embeddings(x_week) #(*,embedding_size)        
#             x_ = torch.cat([ x_emb, x_features], 1)
#         x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
#         out, x_markov.unsqueeze(-1)], -1)
#         n_f = x_.shape[-1]
#         x_ =  self.fc2(x_).squeeze(-1)

#         if self.decision_focused:
#             mlog = nn.LogSoftmax(dim=1)
#             return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
#         else:
#             mlog = nn.LogSoftmax(dim=1)
#             return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)
# class NoMarkov(MarkovLinear):
#     def __init__(self, embedding_size,
#         lookback_period, target_dim=1,nnodes= 74,n_features=2, 
#         nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
#         decision_focused=False, **kwargs):
#         super().__init__(embedding_size,
#         lookback_period, target_dim,nnodes,n_features, 
#         nweekdays,onehot, drop_prob,weekly,
#         decision_focused, **kwargs)


#         if weekly:
#             self.fc2 = nn.Linear(n_features+2,1)       
#         elif onehot:
#             self.fc2 = nn.Linear(nweekdays+n_features+2,1)
#         else:
#             self.fc2 = nn.Linear(embedding_size+n_features+2,1)
#             self.embeddings = nn.Embedding(nweekdays, embedding_size)        


#     def forward(self,x, x_dist, x_features,x_markov, x_week,x_mask):
#         n_rows = x_features.shape[0]
#         m = nn.Softmax(dim=1)
#         out = self.fc1(x.transpose(1,2))
#         if self.weekly:
#             x_ = x_features
#         else:
#             if self.onehot:
#                 x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
#             else:
#                 x_emb = self.embeddings(x_week) #(*,embedding_size)        
#             x_ = torch.cat([ x_emb, x_features], 1)
#         x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
#         out,x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1) ], -1)
#         n_f = x_.shape[-1]
#         x_ =  self.fc2(x_).squeeze(-1)

#         if self.decision_focused:
#             mlog = nn.LogSoftmax(dim=1)
#             return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
#         else:
#             mlog = nn.LogSoftmax(dim=1)
#             return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)

class NoMarkov(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, stop_embedding_size=12,target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p= drop_prob)        
        self.fc1 = nn.Linear(lookback_period,1)
        if weekly:
            self.fc2 = nn.Linear(n_features+2+stop_embedding_size,1)       
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features+2+stop_embedding_size,1)
        else:
            self.fc2 = nn.Linear(embedding_size+n_features+2+stop_embedding_size,1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)        
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,stops,x, x_dist, x_features,x_markov, x_week,x_mask):
        x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))
        
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2))
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)

        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        x_embed.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        out,x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1)], -1)

        n_f = x_.shape[-1]
        x_ =  self.fc2(x_).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)

class NoFeatures(MarkovLinear):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__(embedding_size,
        lookback_period, target_dim,nnodes,n_features, 
        nweekdays,onehot, drop_prob,weekly,
        decision_focused, **kwargs)


        if weekly:
            self.fc2 = nn.Linear(3,1)       
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+3,1)
        else:
            self.fc2 = nn.Linear(embedding_size+3,1)
            self.embeddings = nn.Embedding(nweekdays, embedding_size)        


    def forward(self,x, x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2))
        if self.weekly:
            x_ = 0
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = x_emb
        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        out,x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)], -1)
        n_f = x_.shape[-1]
        x_ =  self.fc2(x_).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)

class NoWeek(MarkovLinear):
    def __init__(self, embedding_size,
        lookback_period,stop_embedding_size=12, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__(embedding_size,
        lookback_period, target_dim,nnodes,n_features, 
        nweekdays,onehot, drop_prob,weekly,
        decision_focused, **kwargs)


        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.droput = nn.Dropout(p= drop_prob)        
        self.fc1 = nn.Linear(lookback_period,1)
  
        self.stop_embeddings = nn.Embedding(nnodes, stop_embedding_size)  



        self.fc2 = nn.Linear(n_features+3+stop_embedding_size,1) 
      
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,stops,x, x_dist, x_features,x_markov, x_week,x_mask):
        x_embed = torch.stack( tuple( self.stop_embeddings(torch.LongTensor(s)).sum(0) for s in stops))
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2))
        x_ = x_features

        x_ = torch.cat([ x_.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        x_embed.unsqueeze(1).expand(n_rows,self.nnodes,-1),
        out,x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)], -1)  

        n_f = x_.shape[-1]
        x_ =  self.fc2(x_).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)

class OnlyMarkov(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        self.linear = nn.Linear(1,1)

        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,stops,x, x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        x_ = self.linear( x_markov.unsqueeze(-1)).squeeze(-1)
    
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            # return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)
            # return m(x_)


class OnlyMarkovConvolution(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(1,nnodes,1)

        self.weekly= weekly        
        
        self.nweekdays = nweekdays
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,x, x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.conv( x_markov.unsqueeze(1))
        x_ = torch.diagonal(out,  dim1=1, dim2=2)
    
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            # return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)
            # return m(x_)



class AdditiveResidualpred(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False,**kwargs):
        super().__init__()

        self.weekly= weekly        
      
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features,nnodes)          
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features,nnodes)
        
        self.fc1 = nn.Linear(lookback_period,1)
        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused = decision_focused
    def forward(self,x,  x_dist, x_features,x_week,x_mask):
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)   
        x_ = (self.fc2(x_)+ out - x_dist)
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)

class AdditiveFullpred(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + nnodes,nnodes)          
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features + nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,x,  x_dist, x_features,x_week,x_mask):
        n_rows = x_features.shape[0]

        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)
      
        if self.weekly:
            x_ = torch.cat([  x_features, x_dist.expand(n_rows,-1)], 1)
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features, x_dist.expand(n_rows,-1)], 1)    

        x_ = (self.fc2(x_)+ out)
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)

class AdditiveFullpred2(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused= False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + 2,1)          
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features + 2,1)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + 2,1)

        self.fc1 = nn.Linear(lookback_period,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused = decision_focused

    def forward(self,x,  x_dist, x_features,x_week,x_mask):
        n_rows = x_features.shape[0]

        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)) # (*,74)

        if self.weekly:
            x_ = x_features.unsqueeze(1).expand(-1,self.nnodes,-1 ) 
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 
            1).unsqueeze(1).expand(-1,self.nnodes,-1 )
        
        x_dist_ = x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1)

        x_ = (self.fc2(torch.cat([ x_dist_, out,x_], 2) )).squeeze(-1)
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)

class AdditiveFullpred3(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features +2*nnodes,nnodes)   
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features +2*nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + 2*nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused = decision_focused

    def forward(self,x,  x_dist, x_features,x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)
      
        x_ = self.fc2(torch.cat([ x_dist.expand(n_rows,-1), out,x_], 1) )

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)


class AdditiveResidualpred3(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + nnodes,nnodes)        
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features + nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,x,  x_dist, x_features,x_week,x_mask):
        n_rows = x_features.shape[0]


        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)
               
        x_ = self.fc2(torch.cat([  out,x_], 1) ) -  x_dist

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)


class AdditiveResidualpred4(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + nnodes,nnodes)        
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features + nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)
        self.fc3 = nn.Linear(2 ,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,x,  x_dist, x_features,x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)     
        x_ = self.fc2(torch.cat([  out,x_], 1) )
        x_ = (self.fc3(torch.cat([x_.unsqueeze(-1),
        x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1)],2))).squeeze(-1)
        
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            # return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)
            # return m(x_)


class AdditiveMarkovpred(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()
        

        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + 3*nnodes,nnodes)
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features + 3*nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + 3*nnodes,nnodes)
        self.fc1 = nn.Linear(lookback_period,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.weekly = weekly
        self.decision_focused = decision_focused

    def forward(self,x,  x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]

        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)
        
        x_ = self.fc2(torch.cat([ x_dist.expand(n_rows,-1), out,x_, x_markov], 1) )
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)



class ResidualMarkovpred(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly = False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + 2*nnodes,nnodes)        
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features + 2*nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + 2*nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused = decision_focused

    def forward(self,x,  x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]

        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)

        
        x_ = self.fc2(torch.cat([  out,x_, x_markov], 1) ) - x_dist
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)


class MarkovConjunctionpred(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly= False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + 2*nnodes,nnodes)
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features +2*nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + 2*nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)
        self.fc3 = nn.Linear(2*nnodes,nnodes)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,x,  x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)

        x_ = self.fc2(torch.cat([ x_dist.expand(n_rows,-1), out,x_], 1) )
        x_ = self.fc3(torch.cat([x_,x_markov],1))
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)


class MarkovConjunctionpred2(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + 2*nnodes,nnodes)        
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features +2*nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + 2*nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)
        self.fc3 = nn.Linear(2 ,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused = decision_focused

    def forward(self,x,  x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)

        x_ = self.fc2(torch.cat([ x_dist.expand(n_rows,-1), out,x_], 1) )
        x_ = self.fc3(torch.cat([x_.unsqueeze(-1),
        x_markov.unsqueeze(-1)],2)).squeeze(-1)
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)


class MarkovConjunctionpred3(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + nnodes,nnodes)        
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features +nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)
        self.fc3 = nn.Linear(3 ,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused =decision_focused

    def forward(self,x, x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)

        x_ = self.fc2(torch.cat([ out,x_], 1) )
        x_ = self.fc3(torch.cat([x_.unsqueeze(-1),
        x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)],2)).squeeze(-1)
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
        else:
            # return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)
            mlog = nn.LogSoftmax(dim=1)
            return (mlog(x_)).masked_fill(x_mask.bool(),-1e8)
            # return m(x_)


class MarkovConjunctionpred4(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features + nnodes,nnodes)        
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features +nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features + nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)
        self.fc3 = nn.Linear(2 ,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused = decision_focused

    def forward(self,x,  x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)     
        x_ = self.fc2(torch.cat([  out,x_], 1) )
        x_ = (self.fc3(torch.cat([x_.unsqueeze(-1),
        x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1)],2)) +
         x_markov.unsqueeze(-1)).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)


class MarkovConjunctionpred5(nn.Module):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__()

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear(n_features ,1)        
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features,1)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features ,1)

        self.fc1 = nn.Linear(lookback_period,1)
        self.fc3 = nn.Linear(4 ,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused = decision_focused

    def forward(self,x,  x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)) # (*,74)

        if self.weekly:
            x_ = self.fc2(x_features).unsqueeze(1).expand(-1,self.nnodes,-1)
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = self.fc2(torch.cat([ x_emb, x_features], 
                    1)).unsqueeze(1).expand(-1,self.nnodes,-1)

        x_ = self.fc3(torch.cat([out, x_,
        x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
         x_markov.unsqueeze(-1)],2)).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)

class MarkovNoFeatures(MarkovConjunctionpred4):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__(embedding_size,
        lookback_period, target_dim,nnodes,n_features, 
        nweekdays,onehot, drop_prob,weekly,
        decision_focused, **kwargs)

        self.weekly= weekly        
        
        self.nweekdays = nweekdays

        self.droput = nn.Dropout(p= drop_prob)
        if weekly:
            self.fc2 = nn.Linear( nnodes,nnodes)        
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+nnodes,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+ nnodes,nnodes)

        self.fc1 = nn.Linear(lookback_period,1)
        self.fc3 = nn.Linear(2 ,1)

        self.nnodes = nnodes
        self.onehot = onehot
        self.lookback_period =lookback_period
        self.decision_focused = decision_focused

    def forward(self,x,  x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)
        if self.weekly:
            x_ = self.fc2(torch.cat([  out], 1) )
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            # x_ = torch.cat([ x_emb, x_features], 1)   
            x_ = self.fc2(torch.cat([  out,x_emb], 1) )  
        # x_ = self.fc2(torch.cat([  out,x_], 1) )
        x_ = (self.fc3(torch.cat([x_.unsqueeze(-1),
        x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1)],2)) +
         x_markov.unsqueeze(-1)).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)



class MarkovNoPast(MarkovConjunctionpred3):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False, **kwargs):
        super().__init__(embedding_size,
        lookback_period, target_dim,nnodes,n_features, 
        nweekdays,onehot, drop_prob,weekly,
        decision_focused, **kwargs)


        if weekly:
            self.fc2 = nn.Linear(n_features ,nnodes)        
        elif onehot:
            self.fc2 = nn.Linear(nweekdays+n_features ,nnodes)
        else:
            self.embeddings = nn.Embedding(nweekdays, embedding_size)
            self.fc2 = nn.Linear(embedding_size+n_features ,nnodes)


    def forward(self,x, x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)

        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)

        x_ = self.fc2(x_ )
        x_ = self.fc3(torch.cat([x_.unsqueeze(-1),
        x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1),
        x_markov.unsqueeze(-1)],2)).squeeze(-1)
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)

##### Expnetial Smoothing in Neural Net models

class MarkovConjunctionpred4Smoother(MarkovConjunctionpred4):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False,case=1,relu=True, **kwargs):
        super().__init__(embedding_size,
        lookback_period, target_dim,nnodes,n_features, 
        nweekdays,onehot, drop_prob,weekly,
        decision_focused, **kwargs)
        #####      added for smoothing ######
        w = nn.Parameter(torch.ones(1)*0.1)
        self.register_parameter(name='smoother', param=w)
        self.fc1 = nn.Linear(1,1)    
        self.case, self.relu = case, relu  
        #####################################  

    def forward(self,x,  x_dist, x_features,x_markov, x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        ###############
        reluop = nn.ReLU()
        wt = torch.ones(self.lookback_period)
        e = torch.arange(self.lookback_period)
        if self.case==1:
            multiplier = torch.pow(torch.sigmoid(self.smoother)*wt, e) # 1. a^t
            
        if self.case==2:
            # 2. (beta0 + beta1 a)^t
            if self.relu:
                multiplier = torch.pow(reluop(self.fc1(torch.sigmoid(self.smoother)))*wt,
                 e)
            else:
                multiplier = torch.pow(self.fc1(torch.sigmoid(self.smoother))*wt, e)

        if self.case==3:
            # 3. (beta0 + beta1 a^t)
            if self.relu:
                multiplier = reluop(self.fc1(torch.pow(torch.sigmoid(self.smoother)*wt, 
                e).view(-1,1)).squeeze())  
            else:
                multiplier = self.fc1(torch.pow(torch.sigmoid(self.smoother)*wt, 
                e).view(-1,1)).squeeze()

        if self.case==4:
            # 4. (beta0 + beta1 0.7^t)
            if self.relu:
                multiplier = reluop(self.fc1(torch.pow(0.7*wt, e).view(-1,1)).squeeze())  
            else:
                multiplier = self.fc1(torch.pow(0.7*wt, e).view(-1,1)).squeeze()
        out  = (x.transpose(1,2)*multiplier).sum(2)
        ###############
        # out = self.fc1(x.transpose(1,2)).squeeze(-1)
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)     
        x_ = self.fc2(torch.cat([  out,x_], 1) )
        x_ = (self.fc3(torch.cat([x_.unsqueeze(-1),
        x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1)],2)) +
         x_markov.unsqueeze(-1)).squeeze(-1)

        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)

class AdditiveResidualpred4Smoother(AdditiveResidualpred4):
    def __init__(self, embedding_size,
        lookback_period, target_dim=1,nnodes= 74,n_features=2, 
        nweekdays=7,onehot=False, drop_prob=0.01,weekly=False,
        decision_focused=False,case=1,relu=True, **kwargs):

        super().__init__(embedding_size,
        lookback_period, target_dim,nnodes,n_features, 
        nweekdays,onehot, drop_prob,weekly,
        decision_focused, **kwargs)
        #####      added for smoothing ######
        w = nn.Parameter(torch.ones(1)*0.1)
        self.register_parameter(name='smoother', param=w)
        self.fc1 = nn.Linear(1,1)    
        self.case, self.relu = case, relu  
        ##################################### 

    def forward(self,x,  x_dist, x_features,x_week,x_mask):
        n_rows = x_features.shape[0]
        m = nn.Softmax(dim=1)
        ###############
        reluop = nn.ReLU()
        wt = torch.ones(self.lookback_period)
        e = torch.arange(self.lookback_period)
        if self.case==1:
            multiplier = torch.pow(torch.sigmoid(self.smoother)*wt, e) # 1. a^t
            
        if self.case==2:
            # 2. (beta0 + beta1 a)^t
            if self.relu:
                multiplier = torch.pow(reluop(self.fc1(torch.sigmoid(self.smoother)))*wt,
                 e)
            else:
                multiplier = torch.pow(self.fc1(torch.sigmoid(self.smoother))*wt, e)

        if self.case==3:
            # 3. (beta0 + beta1 a^t)
            if self.relu:
                multiplier = reluop(self.fc1(torch.pow(torch.sigmoid(self.smoother)*wt, 
                e).view(-1,1)).squeeze())  
            else:
                multiplier = self.fc1(torch.pow(torch.sigmoid(self.smoother)*wt, 
                e).view(-1,1)).squeeze()

        if self.case==4:
            # 4. (beta0 + beta1 0.7^t)
            if self.relu:
                multiplier = reluop(self.fc1(torch.pow(0.7*wt, e).view(-1,1)).squeeze())  
            else:
                multiplier = self.fc1(torch.pow(0.7*wt, e).view(-1,1)).squeeze()
        out  = (x.transpose(1,2)*multiplier).sum(2)
        ###############

        # out = self.fc1(x.transpose(1,2)).squeeze(-1) # (*,74)
        if self.weekly:
            x_ = x_features
        else:
            if self.onehot:
                x_emb =  make_onehot(x_week,self.nweekdays) # (*,nweekdays)
            else:
                x_emb = self.embeddings(x_week) #(*,embedding_size)        
            x_ = torch.cat([ x_emb, x_features], 1)     
        x_ = self.fc2(torch.cat([  out,x_], 1) )
        x_ = (self.fc3(torch.cat([x_.unsqueeze(-1),
        x_dist.unsqueeze(-1).unsqueeze(0).expand(n_rows,-1,1)],2))).squeeze(-1)
        
        if self.decision_focused:
            mlog = nn.LogSoftmax(dim=1)
            return (-mlog(x_)).masked_fill(x_mask.bool(),1e8)
            # return m(x_)
        else:
            return f.normalize(m(x_).masked_fill(x_mask.bool(),0), p=1, dim=1)



class LSTMonlypred(nn.Module):
    def __init__(self,input_dim, hidden_size, num_layers, target_dim,
                 embedding_size,lookback_period, nnodes= 74, 
                 weekly=False,nweekdays=7,drop_prob=0.2,
                 **kwargs):
        # super(LSTMNet, self).__init__()
        super().__init__()
        self.weekly = weekly
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim,hidden_size, num_layers, 
        	batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size,target_dim)
        self.relu = nn.ReLU()
        self.lookback_period =lookback_period
        
    def forward(self, x, h,x_dist, x_features,x_week,x_mask):
        out, h = self.lstm(x, h)
        out = self.fc(out[:,-1,:])
        return out* x_mask

class VRPLSTMNet:
    def __init__(self,training_stops,num_layers,
                 net,  embedding_size =2, 
                 nnodes=74, nweekdays=7,lookback_period=30,n_features=2,
                 optimizer=optim.Adam,epochs=20,**kwargs):
        self.training_stops = training_stops
        self.net = net
        
        self.nnodes = nnodes
        self.lookback_period = lookback_period
        self.epochs = epochs

       
        self.model = self.net(nnodes,nnodes,num_layers,nnodes,lookback_period=lookback_period,
        embedding_size= embedding_size,nnodes=nnodes,n_features=n_features,
         nweekdays=7)
         
        optim_args= [k for k, v in signature(optimizer).parameters.items()]
        optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}

        self.optimizer = optimizer(self.model.parameters(), **optim_dict)
    def fit(self,trgt,weekday,stops_list,n_vehicleslist,distance_mat, *arg,**kwrg):
        '''
        x_train : past data: start:[0: (N -lookback_period )], 
                             end: [ lookback_period -1 : N-1 ]
        x_features: features of that day: [ lookback_period : N] [n_features]
        y: trgt of that day - [ lookback_period : N]
        '''

        training_days = len(trgt) - self.lookback_period
        xdist = log_softmax(distance_mat[self.training_stops]).astype(np.float32)

        x_train = np.zeros((training_days,
        	self.lookback_period,self.nnodes)).astype(np.float32)
        y_train = np.zeros((training_days,
        self.nnodes)).astype(np.float32)

        x_mask = np.ones((training_days,self.nnodes)).astype(np.float32)
        x_vehicle = np.zeros((training_days,1)).astype(np.float32)
        x_stops = np.zeros((training_days,1)).astype(np.float32)
        x_week = np.zeros(training_days).astype(np.int64)
        cnt = 0
        for d in range(training_days):
            lacti = stops_list[d + self.lookback_period]
            if self.training_stops not in lacti:
                raise Exception('Stop should be in the training days')
                # print("Stops {} Not in day {}".format(self.training_stops,
                #     d))
            else:
                x_vehicle[d] = n_vehicleslist[d + self.lookback_period]
                x_stops[d] = len(lacti)
                x_week[d] = weekday[d + self.lookback_period]
                x_mask[d][lacti] = 0
                x_train[d] = trgt[d:(d+ self.lookback_period),self.training_stops,:]
                y_train[d] = trgt[d + self.lookback_period][self.training_stops]
                cnt +=1        

        
        x_mask =  torch.from_numpy(x_mask[:cnt]).to(device)
        self.scaler = MinMaxScaler()
        x_stops = self.scaler.fit_transform(x_stops)
        x_features = np.concatenate((x_stops, x_vehicle), axis=1)
        # x_features = np.zeros((training_days,0)).astype(np.float32)
        x_features = torch.from_numpy(x_features[:cnt]).to(device)  

        x_week = torch.from_numpy(x_week[:cnt]).to(device)
        xdist = torch.from_numpy(xdist).to(device)

        x_train = torch.from_numpy(x_train[:cnt]).type(torch.Tensor)
        y_train = torch.from_numpy(y_train[:cnt]).type(torch.Tensor)

        criterion = nn.BCELoss() #nn.BCEWithLogitsLoss()
        for ep in range(self.epochs):
            self.optimizer.zero_grad()
            op = self.model(x_train,
            xdist, x_features,x_week,x_mask)
            # loss = criterion(op, y_train)
            loss = -(op  * y_train).sum()/len(y_train)
            loss.backward()
            # print("Epochs: {} Loss: {}".format(ep,loss.item()))
            self.optimizer.step()
    
    
    def predict(self,distance_mat,stops_list,weekday,n_vehicleslist,past,proba=True,*arg,**kwrg):
        """
        predict for 1 day
        past : till the beginning but only days which self.training stops is on
        
        """
        
        
        # assert len(stops_list)== len(weekday) == len(n_vehicleslist) ==1
        evaluation_days = len(stops_list) 
        assert len(past) >= evaluation_days +  self.lookback_period-1


        logging.info("evaluation days {}".format(evaluation_days))
        x_evaluation = np.zeros((evaluation_days,
                                self.lookback_period,self.nnodes)).astype(np.float32)
        
        y = np.zeros((evaluation_days,self.nnodes)).astype(np.float32)
        x_mask = np.ones((evaluation_days ,self.nnodes)).astype(np.float32)
        x_vehicle = np.zeros((evaluation_days,1)).astype(np.float32)
        x_stops = np.zeros((evaluation_days,1)).astype(np.float32)
        x_week = np.zeros(evaluation_days).astype(np.int64)
        xdist = log_softmax(distance_mat[self.training_stops]).astype(np.float32) 
        
        cnt = 0
        active_days =[]
        for d in range(evaluation_days):
            lacti = stops_list[d ]

            if self.training_stops not in lacti:
                warnings.warn('Stop {} not in the evaluation days, \n predicted probabilities would be zero'.format(self.training_stops))
                # print("Stops {} Not in day {}".format(self.training_stops,d))
            else:
                if d==(evaluation_days-1):
                    x_evaluation[d] = past[-(self.lookback_period+evaluation_days -d-1):,
                    self.training_stops,:]
                else:
                    x_evaluation[d] = past[-(self.lookback_period+evaluation_days -d-1):-(evaluation_days -d-1),
                    self.training_stops,:]

                x_vehicle[d] = n_vehicleslist[d ]
                x_stops[d] = len(lacti)
                x_week[d] = weekday[d ]
                x_mask[d][lacti] = 0
                active_days.append(d)
                cnt +=1
            # y_evaluation[d]  = trgt[(d+ self.lookback_period),self.training_stops,:]
        x_evaluation = torch.from_numpy(x_evaluation[:cnt]).type(torch.Tensor)
        x_mask =  torch.from_numpy(x_mask[:cnt]).to(device)
        x_stops = self.scaler.transform(x_stops)
        x_features = np.concatenate((x_stops, x_vehicle), axis=1)
       

        x_features = torch.from_numpy(x_features[:cnt]).to(device)         
        # x_vehicle = torch.from_numpy(x_vehicle[:cnt]).to(device)
        x_week = torch.from_numpy(x_week[:cnt]).to(device)
        xdist = torch.from_numpy(xdist).to(device)

        self.model.eval()
        logging.info("-----Trained Model Parameters----")
        logging.info("Model name {}".format(self.net))

        for mp in self.model.parameters():
            logging.info("{} {} {}".format(mp.name, mp.data,mp.shape))
        logging.info("----- --------- ----")

        

        op = self.model (x_evaluation,
        xdist, x_features,x_week,x_mask)
        # m = nn.Softmax(dim=1)
        # op = m(op)
        
        self.model.train()
        y[active_days] = op.detach().numpy()
        if not proba:
            y = torch.argmax(y,dim=1)
        return y


# class Additivepred(nn.Module):
#     def __init__(self, embedding_size,
#         lookback_period, target_dim=1,
#         nnodes= 74,n_features=2, nweekdays=7,drop_prob=0.01):

class VRPAdditiveHistory(VRPLSTMNet):
    def __init__(self,training_stops,onehot=False,
                 net= AdditiveResidualpred, num_layers=2,embedding_size =2, 
                 nnodes=74, nweekdays=7,lookback_period=30,n_features=2,
                 optimizer=optim.Adam,epochs=20,case=1, relu=False, **kwargs):
        
        self.training_stops = training_stops
        self.net = net
        self.num_layers = num_layers
        self.nnodes = nnodes
        self.lookback_period = lookback_period
        self.epochs = epochs
        
        
        self.model = self.net(
         embedding_size,lookback_period,
        nnodes=nnodes,n_features=n_features, 
        nweekdays=7,onehot= onehot,case=case,relu=relu)

        optim_args= [k for k, v in signature(optimizer).parameters.items()]
        optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}

        self.optimizer = optimizer(self.model.parameters(), **optim_dict)

class Neural_and_Markov:
    def __init__(self,training_stops,onehot=False,
                 net= AdditiveMarkovpred,  num_layers=2,embedding_size =2, 
                 nnodes=74, nweekdays=7,lookback_period=30,n_features=2,
                 optimizer=optim.Adam,epochs=20,weekly=False, **kwargs):
        self.training_stops = training_stops
        self.net = net
 
        self.nnodes = nnodes
        self.lookback_period = lookback_period
        self.epochs = epochs
        self.weekly = weekly

       
        self.model = self.net(
         embedding_size,lookback_period,
        nnodes=nnodes,n_features=n_features, 
        nweekdays=7,onehot= onehot, weekly=weekly)
         
        optim_args= [k for k, v in signature(optimizer).parameters.items()]
        optim_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in optim_args}

        self.optimizer = optimizer(self.model.parameters(), **optim_dict)
    def fit(self,trgt,weekday,stops_list,n_vehicleslist,distance_mat, *arg,**kwrg):
        '''
        x_train : past data: start:[0: (N -lookback_period )], 
                             end: [ lookback_period -1 : N-1 ]
        x_features: features of that day: [ lookback_period : N] [n_features]
        y: trgt of that day - [ lookback_period : N]
        '''

        training_days = len(trgt) - self.lookback_period
        xdist = log_softmax(distance_mat[self.training_stops]).astype(np.float32)

        x_train = np.zeros((training_days,
        	self.lookback_period,self.nnodes)).astype(np.float32)
        y_train = np.zeros((training_days,
        self.nnodes)).astype(np.float32)

        x_mask = np.ones((training_days,self.nnodes)).astype(np.float32)
        x_vehicle = np.zeros((training_days,1)).astype(np.float32)
        x_stops = np.zeros((training_days,1)).astype(np.float32)
        x_week = np.zeros(training_days).astype(np.int64)

        x_markov = np.zeros((training_days,
        self.nnodes)).astype(np.float32)
        cnt = 0
        for d in range(training_days):
            lacti = stops_list[d + self.lookback_period]
            if self.training_stops not in lacti:
                raise Exception('Stop should be in the training days')
                # print("Stops {} Not in day {}".format(self.training_stops,
                #     d))
            else:
                markovmodel = MarkovCounter(self.training_stops,
                exp=0.7,beta=1,smoothing_value= 0.1,weekly= True)
                markovmodel.fit(trgt[:(d + self.lookback_period)],
                weekday[:(d + self.lookback_period)])
                x_markov[d] = markovmodel.predict(distance_mat,
                 stops_list[[(d + self.lookback_period)]],
                 weekday[[(d + self.lookback_period)]],
                 n_vehicleslist[[(d + self.lookback_period)]],
                 trgt[:(d + self.lookback_period+1)])

                x_vehicle[d] = n_vehicleslist[d + self.lookback_period]
                x_stops[d] = len(lacti)
                x_week[d] = weekday[d + self.lookback_period]
                x_mask[d][lacti] = 0
                x_train[d] = trgt[d:(d+ self.lookback_period),self.training_stops,:]
                y_train[d] = trgt[d + self.lookback_period][self.training_stops]
                cnt +=1  

        self.final_markovmodel =   markovmodel    

        
        x_mask =  torch.from_numpy(x_mask[:cnt]).to(device)
        self.scaler = MinMaxScaler()
        x_stops = self.scaler.fit_transform(x_stops)
        x_features = np.concatenate((x_stops, x_vehicle), axis=1)
        # x_features = np.zeros((training_days,0)).astype(np.float32)
     

        x_features = torch.from_numpy(x_features[:cnt]).to(device)
        x_markov = torch.from_numpy(x_markov[:cnt]).to(device)  

        x_week = torch.from_numpy(x_week[:cnt]).to(device)
        xdist = torch.from_numpy(xdist).to(device)

        x_train = torch.from_numpy(x_train[:cnt]).type(torch.Tensor)
        y_train = torch.from_numpy(y_train[:cnt]).type(torch.Tensor)

        criterion = nn.BCELoss() #nn.BCEWithLogitsLoss()
        for ep in range(self.epochs):
            self.optimizer.zero_grad()
        
            op = self.model(x_train,
            xdist, x_features,x_markov, x_week,x_mask)
            # loss = criterion(op, y_train)
            loss = -(op  * y_train).sum()/len(y_train)
            loss.backward()
            # print("Epochs: {} Loss: {}".format(ep,loss.item()))
            self.optimizer.step()
    
    
    def predict(self,distance_mat,stops_list,weekday,n_vehicleslist,past,proba=True,*arg,**kwrg):
        """
        predict for 1 day
        stops_list : for day t
        past : since the beginning till t 
               but only days which self.training stops is on
        
        """
        
        
        # assert len(stops_list)== len(weekday) == len(n_vehicleslist) ==1
        evaluation_days = len(stops_list) 
        assert len(past) >= evaluation_days +  self.lookback_period-1


        logging.info("evaluation days {}".format(evaluation_days))
        x_evaluation = np.zeros((evaluation_days,
                                self.lookback_period,self.nnodes)).astype(np.float32)
        
        y = np.zeros((evaluation_days,self.nnodes)).astype(np.float32)
        x_mask = np.ones((evaluation_days ,self.nnodes)).astype(np.float32)
        x_vehicle = np.zeros((evaluation_days,1)).astype(np.float32)
        x_stops = np.zeros((evaluation_days,1)).astype(np.float32)
        x_week = np.zeros(evaluation_days).astype(np.int64)
        xdist = log_softmax(distance_mat[self.training_stops]).astype(np.float32) 
        x_markov = np.zeros((evaluation_days,self.nnodes)).astype(np.float32)
        
        cnt = 0
        active_days =[]
        for d in range(evaluation_days):
            lacti = stops_list[d ]

            if self.training_stops not in lacti:
                warnings.warn('Stop {} not in the evaluation days, \n predicted probabilities would be zero'.format(self.training_stops))
                # print("Stops {} Not in day {}".format(self.training_stops,d))
            else:
                if d==(evaluation_days-1):
                    x_evaluation[d] = past[-(self.lookback_period+evaluation_days -d-1):,
                    self.training_stops,:]
                else:
                    x_evaluation[d] = past[-(self.lookback_period+evaluation_days -d-1):-(evaluation_days -d-1),
                    self.training_stops,:]


                x_markov[d] = self.final_markovmodel.predict(distance_mat,
                 stops_list[[d]],weekday[[d]], n_vehicleslist[[d]], past)


                x_vehicle[d] = n_vehicleslist[d ]
                x_stops[d] = len(lacti)
                x_week[d] = weekday[d ]
                x_mask[d][lacti] = 0
                active_days.append(d)
                cnt +=1
            # y_evaluation[d]  = trgt[(d+ self.lookback_period),self.training_stops,:]
        x_evaluation = torch.from_numpy(x_evaluation[:cnt]).type(torch.Tensor)
        x_mask =  torch.from_numpy(x_mask[:cnt]).to(device)
        x_stops = self.scaler.transform(x_stops)
        x_features = np.concatenate((x_stops, x_vehicle), axis=1)
       

        x_features = torch.from_numpy(x_features[:cnt]).to(device)         
        # x_vehicle = torch.from_numpy(x_vehicle[:cnt]).to(device)
        x_week = torch.from_numpy(x_week[:cnt]).to(device)
        xdist = torch.from_numpy(xdist).to(device)
        x_markov = torch.from_numpy(x_markov[:cnt]).to(device)

        self.model.eval()
        logging.info("-----Trained Model Parameters----")
        logging.info("Model name {}".format(self.net))

        for mp in self.model.parameters():
            logging.info("{} {} {}".format(mp.name, mp.data,mp.shape))
        logging.info("----- --------- ----")

        
        op = self.model (x_evaluation,
        xdist, x_features,x_markov, x_week,x_mask)

        
        self.model.train()
        y[active_days] = op.detach().numpy()
        if not proba:
            y = torch.argmax(y,dim=1)
        return y

