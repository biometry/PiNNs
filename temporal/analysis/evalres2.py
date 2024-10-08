# !/usr/bin/env python
# coding: utf-8
# @author: Niklas Moser
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(sys.path)
from misc import HP
from misc import utils
from misc import models
from misc import training
import torch
import pandas as pd
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import csv
import argparse


parser = argparse.ArgumentParser(description='Define data usage')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()

def evalres2(data_use='full'):
    if data_use=='sparse':
        x, y, xt = utils.loaddata('validation', 1, dir="data/", raw=True, sparse=True)
        yp = pd.read_csv("data/hyytialaF_sparse.csv")
        
    else:
        x, y, xt = utils.loaddata('validation', 1, dir="data/", raw=True)
        yp = pd.read_csv("data/hyytialaF_full.csv")
        


    yp.index = pd.DatetimeIndex(yp['date'])
    
    yptr = yp.drop(yp.columns.difference(['GPPp']), axis=1)
    ypte = yp.drop(yp.columns.difference(['GPPp']), axis=1)
    yp_tr = yptr[~yptr.index.year.isin([2004,2005,2007,2008])][1:]
    yp_te = ypte[ypte.index.year==2008][1:]
    y = y.to_frame()
    train_x = x[~x.index.year.isin([2004,2005,2007,2008])]
    train_y = y[~y.index.year.isin([2004,2005,2007,2008])]
    splits = len(train_x.index.year.unique())

    test_x = x[x.index.year == 2008][1:]
    test_y = y[y.index.year == 2008][1:]
    print('TEST X und YP test', train_x, train_y, yp_tr, test_x, test_y, yp_te)
    splits = len(train_x.index.year.unique())
    print(splits)
    train_x.index, train_y.index, yp_tr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(yp_tr)) 
    test_x.index, test_y.index, yp_te.index = np.arange(0, len(test_x)), np.arange(0, len(test_y)), np.arange(0, len(yp_te))

    # Load results from NAS
    d = pd.read_csv(f"temporal/results/Nres2HP_{data_use}.csv")
    a = d.loc[d.ind_mini.idxmin()]
    layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
    parms = np.array(np.matrix(a.parameters)).ravel()
    lr = parms[0]
    bs = int(parms[1])
    model_design = {'layersizes': layersizes}
    print('layersizes', layersizes)

    hp = {'epochs': 5000,
           'batchsize': int(bs),
           'lr': lr
           }

    data_dir = "temporal/models/"
    data = f"res2_{data_use}"
    tloss = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, res=2, ypreles=yp_tr, exp=1)
    train_loss = tloss['train_loss']
    val_loss = tloss['val_loss']
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    for i in range(5000):
        t1.append(train_loss[0][i])
        t2.append(train_loss[1][i])
        t3.append(train_loss[2][i])
        t4.append(train_loss[3][i])
    pd.DataFrame({"f1": t1, "f2": t2, "f3":t3, "f4":t4}).to_csv(f'temporal/results/res2_trainloss_{data_use}.csv')
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    for i in range(5000):
        v1.append(val_loss[0][i])
        v2.append(val_loss[1][i])
        v3.append(val_loss[2][i])
        v4.append(val_loss[3][i])

    pd.DataFrame({"f1": v1, "f2": v2, "f3":v3, "f4":v4}).to_csv(f'temporal/results/res2_vloss_{data_use}.csv')

    # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    x_train, y_train, tr_yp = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32), torch.tensor(yp_tr.to_numpy(), dtype=torch.float32)
    x_test, y_test, te_yp = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32), torch.tensor(yp_te.to_numpy(), dtype=torch.float32)
    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []
    preds_tr = {}
    preds_te = {}
    for i in range(splits):
        i += 1
        #import model
        model = models.RES(x_train.shape[1], y_train.shape[1], model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"res2_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_train = model(x_train, tr_yp)
            p_test = model(x_test, te_yp)
            
            preds_tr.update({f'train_res2{i}':  p_train.flatten().numpy()})
            preds_te.update({f'test_res2{i}':  p_test.flatten().numpy()})

            train_rmse.append(mse(p_train, y_train).tolist())
            train_mae.append(mae(p_train, y_train).tolist())
            test_rmse.append(mse(p_test, y_test).tolist())
            test_mae.append(mae(p_test, y_test).tolist())


    performance = {'train_RMSE': train_rmse,
                    'train_MAE': train_mae,
                    'test_RMSE': test_rmse,
                    'test_mae': test_mae}


    pd.DataFrame.from_dict(performance).to_csv(f'temporal/results/res2_eval_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_tr).to_csv(f'temporal/results/res2_eval_preds_{data_use}_train.csv')
    pd.DataFrame.from_dict(preds_te).to_csv(f'temporal/results/res2_eval_preds_{data_use}_test.csv')


if __name__ == '__main__':
    evalres2(args.d)
