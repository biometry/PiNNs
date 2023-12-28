# !/usr/bin/env python
# coding: utf-8
import utils
import torch
import pandas as pd
import numpy as np
import models
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

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()


def eval2res(data_use='full', of=False, v=2):
    if data_use == 'sparse':
        x, y, xt = utils.loaddata('exp2p', 1, dir="./data/", raw=True, sparse=True, eval=True) #1
    else:
        x, y, xt = utils.loaddata('exp2p', 1, dir="./data/", raw=True, eval=True) #1

    print("OUUTT",x,y,xt)
    # select NAS data
    print(x.index)
    

    train_x = x[((x.index.year == 2005) | (x.index.year == 2008)) & ((x.site_x != "h") & (x.site_y != "h"))]
    train_y = y[((x.index.year == 2005) | (x.index.year == 2008)) & ((x.site_x != "h") & (x.site_y != "h"))]
    
    
    if data_use=="full":
        train_x = train_x.drop(pd.DatetimeIndex(['2005-01-01']))
        train_y = train_y.drop(pd.DatetimeIndex(['2005-01-01']))
    else:
        train_x = train_x.drop(pd.DatetimeIndex(['2005-01-05']))
        train_y = train_y.drop(pd.DatetimeIndex(['2005-01-05']))

    test_x = x[((x.index.year == 2005) | (x.index.year == 2008)) & ((x.site_x == "h") & (x.site_y == "h"))]
    test_y = y[((y.index.year == 2005) | (y.index.year == 2008)) & ((x.site_x == "h") & (x.site_y == "h"))]

    train_x = train_x.drop(['site_x', 'site_y'],axis=1)
    test_x = test_x.drop(['site_x', 'site_y'],axis=1)
    
        
    splits = 4
    print(splits)
    print(x, y)
    train_y = train_y.to_frame()
    test_y = test_y.to_frame()
    train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y))
    if v!=2:
        mlp_as = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2_resAS_{data_use}.csv")
        a = mlp_as.loc[mlp_as.ind_mini.idxmin()][1:5]
        b = a.to_numpy()
        layersizes = list(b[np.isfinite(b)].astype(int))
        print('layersizes', layersizes)
        
        model_design = {'layersizes': layersizes}

        
        mlp_hp = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2_resHP_{data_use}.csv")
        a = mlp_hp.loc[mlp_hp.ind_mini.idxmin()][1:3]
        b = a.to_numpy()
        lr = b[0]
        bs = b[1]
        print(lr)
    else:
        
        d = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2resHP_{data_use}_new.csv")
        a = d.loc[d.ind_mini.idxmin()]
        layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
        parms = np.array(np.matrix(a.parameters)).ravel()
        lr = parms[0]
        bs = int(parms[1])
        model_design = {'layersizes': layersizes}
        print('layersizes', layersizes)                                                                     
    
    if of:
        mlp_hp = pd.read_csv(f"/scratch/project_2000527/pgnn/results/2res_lr_{data_use}.csv")
        a = mlp_hp.loc[mlp_hp.ind_mini.idxmin()][1:3]
        b = a.to_numpy()
        lr = b[0]
        print(lr)
        

    hp = {'epochs': 5000,
          'batchsize': int(bs),
          'lr': lr}
    print('HYPERPARAMETERS', hp)
    data_dir = "./data/"
    data = f"res_{data_use}"
    #print('DATA', train_x, train_y)
    #print('TX', train_x, train_y)
    td, se, ae = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, exp=2)
    print(td, se, ae)
    pd.DataFrame.from_dict(td).to_csv(f'/scratch/project_2000527/pgnn/results/2res_{data_use}_eval_tloss.csv')
    pd.DataFrame.from_dict(se).to_csv(f'/scratch/project_2000527/pgnn/results/2res_{data_use}_eval_vseloss.csv')
    pd.DataFrame.from_dict(ae).to_csv(f'/scratch/project_2000527/pgnn/results/2res_{data_use}_eval_vaeloss.csv')

    # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    x_train, y_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32)
    x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []
    
    preds_train = {}
    preds_test = {}

    for i in range(splits):
        i += 1
        #import model
        model = models.NMLP(test_x.shape[1], 1, model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"2res_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_train = model(x_train)
            p_test = model(x_test)
            preds_train.update({f'train_res{i}': p_train.flatten().numpy()})
            preds_test.update({f'test_res{i}': p_test.flatten().numpy()})
            train_rmse.append(mse(p_train, y_train).tolist())
            train_mae.append(mae(p_train, y_train).tolist())
            test_rmse.append(mse(p_test, y_test).tolist())
            test_mae.append(mae(p_test, y_test).tolist())

    performance = {'train_RMSE': train_rmse,
               'train_MAE': train_mae,
               'test_RMSE': test_rmse,
               'test_mae': test_mae}


    print(preds_train)



    pd.DataFrame.from_dict(performance).to_csv(f'/scratch/project_2000527/pgnn/results/2res_eval_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_train).to_csv(f'/scratch/project_2000527/pgnn/results/2res_eval_preds_{data_use}_train.csv')
    pd.DataFrame.from_dict(preds_test).to_csv(f'/scratch/project_2000527/pgnn/results/2res_eval_preds_{data_use}_test.csv')





if __name__ == '__main__':
    eval2res(args.d)
