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



parser = argparse.ArgumentParser(description='Define data usage')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()

def eval3emb(data_use="full"):

    if data_use == "sparse":
       x,y,xt,yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True, sparse=True, eval=True)
       d1 = ['2005-01-02']
    else:
        x,y,xt,yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True, sparse=False, eval=True)
        d1 = ['2005-01-01']
    y=y.to_frame()
    xt.index = pd.DatetimeIndex(xt['date'])
    print("XT", xt)
    xt = xt.drop(['date', 'year', 'GPPp', 'SWp', 'ETp', 'GPP', 'ET', 'X', 'X.1', 'X.2'], axis=1)[1:]


    train_x = x[(x.index.year == 2005) & (xt.site != "h")]
    train_y = y[(x.index.year == 2005) & (xt.site != "h")]
    train_xt = xt[(xt.index.year == 2005) & (xt.site != "h")]
    train_x = train_x.drop(pd.DatetimeIndex(d1))
    train_xt = train_xt.drop(pd.DatetimeIndex(d1))
    train_y = train_y.drop(pd.DatetimeIndex(d1))

    test_x = x[(x.index.year == 2008) & (xt.site == "h")]
    test_xt = xt[(xt.index.year == 2008) & (xt.site == "h")]
    test_y = y[(y.index.year == 2008) & (xt.site == "h")]

    train_xt= train_xt.drop(['site'],axis=1)
    test_xt = test_xt.drop(['site'],axis=1)
    splits = 4
    layersizes = [[32],[32]]
    model_design = {'layersizes': layersizes}
    print('layersizes', layersizes)


    test_xt.index = np.arange(0, len(test_xt))
    test_x.index = np.arange(0, len(test_x))
    test_y.index = np.arange(0, len(test_y))
    train_x.index= np.arange(0, len(train_x))
    train_y.index= np.arange(0, len(train_y))
    train_xt.index=np.arange(0, len(train_xt))

    train_x = train_x.drop(['site_x', 'site_y'], axis=1)   
    test_x = test_x.drop(['site_x', 'site_y'], axis=1)

    print("TRAIN DATA", train_x, train_y, test_x, test_y)
    batchsize = 366
    lr = 1e-06
    hp = {'epochs': 2,
                          'batchsize': batchsize,
                              'lr': lr}
    data_dir = "../models/"
    data = f"3emb_{data_use}"
    td, se, ae = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=True, hp=False, raw=train_xt, exp=2)
    print(td, se, ae)
    pd.DataFrame.from_dict(td).to_csv(f'./results/3emb_{data_use}_eval_tloss.csv')
    pd.DataFrame.from_dict(se).to_csv(f'./results/3emb_{data_use}_eval_vseloss.csv')
    pd.DataFrame.from_dict(ae).to_csv(f'./results/3emb_{data_use}_eval_vaeloss.csv')


    # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()

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
        model = models.EMB(train_x.shape[1], 1, layersizes, 12, 1)
        model.load_state_dict(torch.load(''.join((data_dir, f"23emb_{data_use}_model{i}.pth"))))
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

    pd.DataFrame.from_dict(performance).to_csv(f'./results/3emb_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_train).to_csv(f'./results/3emb_eval_preds_{data_use}_train.csv')
    pd.DataFrame.from_dict(preds_test).to_csv(f'./results/3emb_eval_preds_{data_use}_test.csv')


if __name__ == "__main__":
    eval3emb(args.d)
