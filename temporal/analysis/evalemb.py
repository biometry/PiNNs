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
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import csv
from sklearn.model_selection import KFold
from torch.autograd import Variable
import argparse


parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()

def evalemb(data_use="full"):

    if data_use == 'sparse':
           x, y, xt = utils.loaddata('validation', 1, dir="../../data/", raw=True, sparse=True)
    else:
           x, y, xt = utils.loaddata('validation', 1, dir="../../data/", raw=True)
    y = y.to_frame()

    xt.index = pd.DatetimeIndex(xt.date)
    xt = xt.drop(['date', 'year', 'GPPp', 'SWp', 'ETp', 'GPP', 'ET'], axis=1)

    train_x = x[~x.index.year.isin([2004, 2005, 2007,2008])]
    train_xt= xt[~xt.index.year.isin([2004, 2005, 2007,2008])][1:]
    train_y = y[~y.index.year.isin([2004, 2005, 2007,2008])]
    splits = len(train_x.index.year.unique())

    test_xt = xt[xt.index.year == 2008]
    test_y = y[y.index.year == 2008]
    test_x = x[x.index.year == 2008]

    test_xt.index = np.arange(0, len(test_xt))
    test_x.index = np.arange(0, len(test_x))
    test_y.index = np.arange(0, len(test_y))
    train_x.index= np.arange(0, len(train_x))
    train_y.index= np.arange(0, len(train_y))
    train_xt.index=np.arange(0, len(train_xt))

    layersizes = [[32],[32]]
    batchsize = 366
    lr = 1e-06
    hp = {'epochs': 5000,
          'batchsize': batchsize,
           'lr': lr, 'eta':1}
    model_design = {'layersizes': layersizes}

    data_dir = "../models/"
    data = f"emb_{data_use}"

    tloss = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=True, hp=False, raw=train_xt)

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
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    for i in range(5000):
        v1.append(val_loss[0][i])
        v2.append(val_loss[1][i])
        v3.append(val_loss[2][i])
        v4.append(val_loss[3][i])

    pd.DataFrame({"f1": v1, "f2": v2, "f3":v3, "f4":v4}).to_csv(f'./results/emb_vloss_{data_use}.csv')
    pd.DataFrame({"f1": t1, "f2": t2, "f3":t3, "f4":t4}).to_csv(f'./results/emb_trainloss_{data_use}.csv')

    # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    train_x = x[~x.index.year.isin([2004, 2005, 2007,2008])]
    train_xt= xt[~xt.index.year.isin([2004, 2005, 2007,2008])][1:]
    train_y = y[~y.index.year.isin([2004, 2005, 2007,2008])]

    test_xt = xt[xt.index.year == 2008]
    test_y = y[y.index.year == 2008]
    test_x = x[x.index.year == 2008]
    test_xt.index = np.arange(0, len(test_xt))
    test_x.index = np.arange(0, len(test_x))
    test_y.index = np.arange(0, len(test_y))
    train_x.index= np.arange(0, len(train_x))
    train_y.index= np.arange(0, len(train_y))
    train_xt.index=np.arange(0, len(train_xt))


    x_train, y_train, xt_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32), torch.tensor(train_xt.to_numpy(), dtype=torch.float32)
    x_test, y_test, xt_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32), torch.tensor(test_xt.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []

    preds_train = {}
    preds_test = {}

    for i in range(splits):
        i += 1
        #import model
        model = models.EMB(x.shape[1], 1, layersizes, 12, 1)
        model.load_state_dict(torch.load(''.join((data_dir, f"emb_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            ypp, p_train = model(x_train, xt_train)
            ypt, p_test = model(x_test, xt_test)
            preds_train.update({f'train_emb{i}': p_train.flatten().numpy()})
            preds_test.update({f'test_emb{i}': p_test.flatten().numpy()})
            train_rmse.append(mse(p_train.flatten(), y_train.flatten()).tolist())
            train_mae.append(mae(p_train.flatten(), y_train.flatten()).tolist())
            test_rmse.append(mse(p_test.flatten(), y_test.flatten()).tolist())
            test_mae.append(mae(p_test.flatten(), y_test.flatten()).tolist())

    performance = {'train_RMSE': train_rmse,
               'train_MAE': train_mae,
               'test_RMSE': test_rmse,
               'test_mae': test_mae}


    print(preds_train)

    pd.DataFrame.from_dict(performance).to_csv(f'./results/emb_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_train).to_csv(f'./results/emb_eval_preds_{data_use}_train.csv')
    pd.DataFrame.from_dict(preds_test).to_csv(f'./results/emb_eval_preds_{data_use}_test.csv')

if __name__ == '__main__':
    evalemb(args.d)

