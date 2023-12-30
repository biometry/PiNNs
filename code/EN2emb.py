# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import torch
import pandas as pd
import numpy as np
import argparse
import HPe
import modelstry
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
import os
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import csv
import training
import argparse
import numpy as np
from sklearn.model_selection import KFold
import os
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
#parser.add_argument('-s', metavar='splits', type=int, help='define number of splits')
args = parser.parse_args()

print(args)

def EN2emb(data_use="full", splits=None, v=2):

    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True)
        #print("DATA----",x,y,xt,yp)
        #x = x.drop(pd.DatetimeIndex(['2004-01-01']))
        #y = y.drop(pd.DatetimeIndex(['2004-01-01']))
    y=y.to_frame()
    xt.index = pd.DatetimeIndex(xt['date'])
    print(xt)
    if data_use=="sparse":
       xt = xt[1:]
    xt = xt[xt.index!='2004-01-01']
    x= x[x.index!="2004-01-01"]
    y= y[y.index!="2004-01-01"]
    xt = xt.drop(['site','X','date', 'year', 'GPPp', 'SWp', 'ETp', 'GPP', 'ET'], axis=1)
    xt.index = np.arange(0, len(xt))
    print("X and Y", x, y, xt)
    print(x.shape, y.shape, xt.shape)
    x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))
    splits=5
    if v==1:
        arch_grid = HP.ArchitectureSearchSpace(x.shape[1], 1, 800, 4)

        # architecture search
        # original: use grid of 800 and epochs:100
        layersizes, argrid = HP.ArchitectureSearch(arch_grid, {'epochs': 200, 'batchsize': 8, 'lr':0.001}, x, y, splits, "arSmlp", hp=True)
        argrid.to_csv(f"/scratch/project_2000527/pgnn/results/NmlpAS_{data_use}.csv")

        # Hyperparameter Search Space
        hpar_grid = HP.HParSearchSpace(800)
    
        # Hyperparameter search
        hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y,  splits, "hpmlp", hp=True)
    
        print( 'hyperparameters: ', hpars)
        grid.to_csv(f"/scratch/project_2000527/pgnn/results/NmlpHP_{data_use}.csv")
        
    elif v==2:
        arch_grid, par_grid = HPe.NASSearchSpace(x.shape[1], 1, 300, 300, 2, emb=True)
        res = HPe.NASSearch(arch_grid, par_grid, x, y, splits, "NASembpar", hp=True, emb=True, exp=2, raw=xt)
        
        df.to_csv(f"N2embHP_{data_use}_new.csv")
                
if __name__ == '__main__':
    EN2emb(args.d)
