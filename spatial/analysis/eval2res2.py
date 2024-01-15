
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
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()


def eval2res2(data_use='full'):
    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True, sparse=True, eval=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True, eval=True)
    # select NAS data
    print(x.index)
    xt.index = pd.DatetimeIndex(xt['date'])
    xt = xt.drop(['date', 'year', 'GPPp', 'SWp', 'ETp', 'GPP', 'ET', 'X'], axis=1)[1:]

    train_x = x[(xt.index.year == 2008) & (xt.site != "h")]
    train_y = y[(xt.index.year == 2008) & (xt.site != "h")]
    
    if data_use=="full":
        train_x = train_x.drop(pd.DatetimeIndex(['2008-01-01']))
        train_y = train_y.drop(pd.DatetimeIndex(['2008-01-01']))
    else:
        train_x = train_x.drop(pd.DatetimeIndex(['2008-01-02']))
        train_y = train_y.drop(pd.DatetimeIndex(['2008-01-02']))

    test_x = x[(xt.index.year == 2008) & (xt.site == "h")]
    test_y = y[(xt.index.year == 2008) & (xt.site == "h")]
    train_x = train_x.drop(['site_x', 'site_y'],axis=1)
    test_x = test_x.drop(['site_x', 'site_y'],axis=1)

    yp.index = xt.index
    yp_tr = yp[(xt.index.year == 2008) & (xt.site != "h")]
    if data_use=="full":
        yp_tr = yp_tr.drop(pd.DatetimeIndex(['2008-01-01']))
    else:
        yp_tr = yp_tr.drop(pd.DatetimeIndex(['2008-01-02']))
    yp_tr = yp_tr.drop(yp.columns.difference(['GPPp']), axis=1)

    yp_te = yp[(xt.index.year == 2008) & (xt.site == "h")]
    yp_te = yp_te.drop(yp.columns.difference(['GPPp']), axis=1)
    splits = 4
    print(splits)
    
    print('test',train_x, train_y, yp_tr)
    train_y = train_y.to_frame()
    test_y = test_y.to_frame()
    train_x.index, train_y.index, yp_tr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(yp_tr))

    d = pd.read_csv(f"../nas/results/N2res2HP_{data_use}.csv")
    a = d.loc[d.ind_mini.idxmin()]
    layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
    parms = np.array(np.matrix(a.parameters)).ravel()
    lr = parms[0]
    bs = int(parms[1])
    model_design = {'layersizes': layersizes}
    print('layersizes', layersizes)



    hp = {'epochs': 5000,
            'batchsize': int(bs),
            'lr': lr}
    print('HYPERPARAMETERS', hp)
    data_dir = "../models/"
    data = f"res2_{data_use}"
    td, se, ae = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, exp=2, res=2, ypreles = yp_tr)
    pd.DataFrame.from_dict(td).to_csv(f'./results/2res2_{data_use}_eval_tloss.csv')
    pd.DataFrame.from_dict(se).to_csv(f'./results/2res2_{data_use}_eval_vseloss.csv')
    pd.DataFrame.from_dict(ae).to_csv(f'./results/2res2_{data_use}_eval_vaeloss.csv')


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
    print(x_train.shape, tr_yp.shape)
    for i in range(splits):
        i += 1
        #import model
        model = models.RES(x_train.shape[1], 1, model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"2res2_{data_use}_model{i}.pth"))))
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


    pd.DataFrame.from_dict(performance).to_csv(f'./results/2res2_eval_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_tr).to_csv(f'./results/2res2_eval_preds_{data_use}_train.csv')
    pd.DataFrame.from_dict(preds_te).to_csv(f'./results/2res2_eval_preds_{data_use}_test.csv')


if __name__ == '__main__':
    eval2res2(args.d)


