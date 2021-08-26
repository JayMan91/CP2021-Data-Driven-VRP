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
from scipy.special import softmax
dtype = torch.float
device = torch.device("cpu")
import logging
from MarkovModel import*
from VRP_model import *
from VRP_LSTMmodel import *



def VRPGurobi(cst_matrix,qcapacity,Q,n_vehicles,activeindices, relaxed=False,gap=1e-2):
    '''
    cst_matrix : n x n matrix
    qcapacity : (n,) array
    Q: Integer
    n_vehicles: Integer
    activeindices: array of stops which should be served, all elements <n

    '''
    
    assert len(cst_matrix) == len(qcapacity)
    n = len(cst_matrix)
    M = Model("vrp")
    M.setParam('OutputFlag', 0)
    M.setParam('MIPGap', 1e-2)
    M.Params.timelimit = 100.0
    vtype= GRB.CONTINUOUS if relaxed else GRB.BINARY
    x = M.addMVar((n,n), vtype=GRB.BINARY)
    u = M.addMVar(n,ub=Q, vtype=GRB.CONTINUOUS)
    M.addConstrs(x[i,i]==0 for i in range(n))
    M.addConstr(u[0]==0)

    M.addConstrs(sum(x[:,i] )==1 for i in activeindices if i>0 )
    M.addConstrs(sum(x[i,:] )==1 for i in activeindices if i>0)
    M.addConstrs(sum(x[:,i] )==0 for i in range(n) if i not in activeindices  )
    M.addConstrs(sum(x[i,:] )==0 for i in range(n) if i not in activeindices  )    
    
    M.addConstr(sum(x[0,:])== n_vehicles ) # hard constraint if capacity is not used
#     M.addConstr(sum(x[:,0]) <= n_vehicles )
    M.addConstrs((x[i,j].tolist()[0] == 1) >> ((u[j].tolist()[0] - u[i].tolist()[0]) == qcapacity[j]) 
                 for i in range(n) for j in range(1,n) if i!=j)
    M.addConstrs(u[i]<=Q for i in range(n))
    M.setObjective( sum(x[:,i] @ cst_matrix[:,i] for i in range(n)), GRB.MINIMIZE)
    M.optimize()
    if M.status==2:
        cmnt = "optimal"
        return True,cmnt, x.X,u.X
    elif M.status==9:
        cmnt = "time-limit"
        return True,cmnt, x.X,u.X
    elif M.status==3:
        cmnt = "INFEASIBLE"
        return False,cmnt, 0,0
    elif M.status==5:
        cmnt = "UNBOUNDED"
        return False,cmnt, 0,0    
    else:
        print("Optimization Status:",M.status)
        cmnt ="Unknown"
        return False,cmnt, 0,0


def VRPsolutiontoList(sol):
    firsts = np.where(sol[0]==1)[0]
    solution = []
    
    for f in range(len(firsts)):
        route = [0]
        # print("Vehicle {}".format(f+1))
        
        dest = 100
        source = firsts[f]
        # print("Vehicle {} goes from Depot to Stop Index {}".format(f+1,source))
        
        while dest !=0:
            route.append(source)
            dest = np.argmax(sol[source])


            # if source==0:
            #     print("Vehicle {} goes from Stop index {} to Stop Index {}".format(f+1,source,dest))
            # elif dest==0:
            #     print("And finally, Vehicle {} goes from Stop index {} to Depot".format(f+1,source))
            # else:
            #     print("Then, Vehicle {} goes  from Stop index {} to Stop Index {} ".format(f+1, source,dest))
            source = dest
        route.append(0)
        solution.append(route)
        # print("______________________")
    return solution

# arc difference
def eval_ad(P, A):
    P_set = set()
    for sublist in P:
        for i in range(len(sublist)-1):
            P_set.add( (sublist[i],sublist[i+1]) )
        
    A_set = set()
    for sublist in A:
        for i in range(len(sublist)-1):
            A_set.add( (sublist[i],sublist[i+1]) )
    
    result = set(A_set).difference((set(P_set)))
    
    # assert(len(A_set) == len(P_set))
    # diffset, diffcount, diffrelative
    return  len(result), len(result)/len(A_set)


# locate minimum
# save location in a list (idx_list)
# before next iteration, increase all values in row and column where minimum is located by a large number
def get_best_route_mapping(P, A):
    # create initial matrix of symmetric differences
    np_matrix = np.zeros((len(P),len(A)))
    for x in range(len(P)):
        for y in range(len(A)):
            np_matrix[x][y] = len(set(P[x]).symmetric_difference(set(A[y])))

    idx_list = []
    while len(idx_list) < len(A):
        # find smallest (x,y) in matrix
        (idx_r, idx_c) = np.where(np_matrix == np.nanmin(np_matrix))
        (r, c) = (idx_r[0], idx_c[0]) # avoid duplicates
        idx_list.append( (r, c) )

        # blank out row/column selected
        np_matrix[r,:] = np.NaN
        np_matrix[:,c] = np.NaN
        #print(np_matrix)
        #print(len(idx_list), len(Act))
    
    return idx_list

# get all unique stops
def allstops(R):
    result = set()
    for route in R:
        result.update(route)
    return result

# stop difference
def eval_sd(P, A):
    # get paired mapping
    idx_list = get_best_route_mapping(P, A)
    
    diff = set()
    for (idx_P, idx_A) in idx_list:
        diff.update(set(P[idx_P]).symmetric_difference(set(A[idx_A])))
    
    nr_stops = len(allstops(A))
    # diffset, diffcount, diffrelative
    return  len(diff), len(diff)/nr_stops