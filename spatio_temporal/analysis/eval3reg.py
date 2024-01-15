# !/usr/bin/env python
# coding: utf-8
# @author: Niklas Moser
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from misc import utils
from misc import models
from misc import training
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
import os
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import csv
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()

def eval3reg(data_use='full', of=False):
    if data_use=='sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True, sparse=True, eval=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True, eval=True)
        
    train_x = x[(x.index.year == 2005) & ((x.site_x != "h") & (x.site_y != "h"))]
    train_y = y[(x.index.year == 2005) & ((x.site_x != "h") & (x.site_y != "h"))]
    if data_use=="full":
        train_x = train_x.drop(pd.DatetimeIndex(['2005-01-01']))
        train_y = train_y.drop(pd.DatetimeIndex(['2005-01-01']))
    else:
        train_x = train_x.drop(pd.DatetimeIndex(['2005-01-05']))
        train_y = train_y.drop(pd.DatetimeIndex(['2005-01-05']))
    test_x = x[(x.index.year == 2008) & ((x.site_x == "h") & (x.site_y == "h"))]
    test_y = y[(y.index.year == 2008) & ((x.site_x == "h") & (x.site_y == "h"))]
    train_x = train_x.drop(['site_x', 'site_y'],axis=1)
    test_x = test_x.drop(['site_x', 'site_y'],axis=1)
    print('TrainTest',train_x, train_y, test_x, test_y)
    yp.index = pd.DatetimeIndex(x.index)
    yp = yp[(x.index.year == 2005) & ((x.site_x != "h") & (x.site_y != "h"))]
        
    if data_use=="full":
        yp = yp.drop(pd.DatetimeIndex(['2005-01-01']))
    else:
        yp = yp.drop(pd.DatetimeIndex(['2005-01-05']))
    reg = yp.drop(yp.columns.difference(['GPPp']), axis=1)

    splits = 4
    
    train_y = train_y.to_frame()
    test_y = test_y.to_frame()
    train_x.index, train_y.index, reg.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(reg))
    print(len(train_x), len(train_y), len(reg))
    
    d = pd.read_csv(f"../../spatial/results/N2regHP_{data_use}.csv")
    a = d.loc[d.ind_mini.idxmin()]
    layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
    parms = np.array(np.matrix(a.parameters)).ravel()
    lr = parms[0]
    bs = int(parms[1])
    eta = parms[2]
    model_design = {'layersizes': layersizes}
    print('layersizes', layersizes)

    hp = {'epochs': 5000,
          'batchsize': int(bs),
      'lr': lr,
        'eta': eta}
    print('HYPERPARAMETERS', hp)
    data_dir = "../models/"
    data = f"3reg_{data_use}"
    #print('DATA', train_x, train_y)
    #print('TX', train_x, train_y)
    td, se, ae = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=reg, emb=False, exp=2)
    print(td, se, ae)
    pd.DataFrame.from_dict(td).to_csv(f'./results/3reg_{data_use}_eval_tloss.csv')
    pd.DataFrame.from_dict(se).to_csv(f'./results/3reg_{data_use}_eval_vseloss.csv')
    pd.DataFrame.from_dict(ae).to_csv(f'./results/3reg_{data_use}_eval_vaeloss.csv')


    mse = nn.MSELoss()
    mae = nn.L1Loss()

    x_train, y_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32)
    x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []

    preds_tr = {}
    preds_te = {}
    for i in range(splits):
        i += 1
        #import model
        model = models.NMLP(x_train.shape[1], y_train.shape[1], model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"23reg_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_train = model(x_train)
            p_test = model(x_test)
            preds_tr.update({f'train_reg{i}':  p_train.flatten().numpy()})
            preds_te.update({f'test_reg{i}':  p_test.flatten().numpy()})
            train_rmse.append(mse(p_train, y_train).tolist())
            train_mae.append(mae(p_train, y_train).tolist())
            test_rmse.append(mse(p_test, y_test).tolist())
            test_mae.append(mae(p_test, y_test).tolist())


    performance = {'train_RMSE': train_rmse,
                   'train_MAE': train_mae,
                   'test_RMSE': test_rmse,
                   'test_mae': test_mae}


    print(performance)


    pd.DataFrame.from_dict(performance).to_csv(f'./results/3reg_eval_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_tr).to_csv(f'./results/3reg_eval_preds_{data_use}_train.csv')
    pd.DataFrame.from_dict(preds_te).to_csv(f'./results/3reg_eval_preds_{data_use}_test.csv')




if __name__=='__main__':
    eval2reg(args.d)
